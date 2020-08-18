import h5py
import pde_superresolution.utils
import pde_superresolution as pde
import json
import numpy as np
import tensorflow as tf
import enum
assert tf.__version__[:2] == '1.'
from matplotlib import pyplot as plt
import xarray
import seaborn
from pde_superresolution import model

@enum.unique
class Dataset(enum.Enum):
  TRAINING = 0
  VALIDATION = 1


def setup_training(dataset, hparams, scale=1.0):
    # predict coefficient as in notebook
    tensors = dataset.make_one_shot_iterator().get_next()
    predictions = predict(tensors['inputs'], hparams)
    loss = tf.reduce_mean((tensors['labels'] - predictions) ** 2) / scale
    train_step = pde.training.create_training_step(loss, hparams)
    return loss, train_step


def baseline_loss(snapshots, hparams):
  dataset = make_dataset(snapshots, hparams, repeat=False, evaluation=True)

  tensors = dataset.make_one_shot_iterator().get_next()
  loss = tf.reduce_mean((tensors['labels'] - tensors['baseline_3']) ** 2)

  sess = tf.Session(config=pde.training._session_config())
  losses = []
  while True:
    try:
      losses.append(sess.run(loss))
    except tf.errors.OutOfRangeError:
      break
  return np.mean(losses)

def predict_coefficients(inputs: tf.Tensor,
                         hparams: tf.contrib.training.HParams,
                         reuse: object = tf.AUTO_REUSE) -> tf.Tensor:
  _, equation = pde.equations.from_hparams(hparams)
  pde.model.assert_consistent_solution(equation, inputs)

  with tf.variable_scope('predict_coefficients', reuse=reuse):
    num_derivatives = len(equation.DERIVATIVE_ORDERS)

    base_grid = pde.polynomials.regular_grid(
        pde.polynomials.GridOffset.STAGGERED, derivative_order=0,
        accuracy_order=hparams.coefficient_grid_min_size,
        dx=1.0)

    net = inputs[:, :, tf.newaxis]
    net /= equation.standard_deviation

    activation = pde.model._NONLINEARITIES[hparams.nonlinearity]

    for _ in range(hparams.num_layers - 1):
      net = pde.layers.conv1d_periodic_layer(net, filters=hparams.filter_size,
                                         kernel_size=hparams.kernel_size,
                                         activation=activation, center=True)

    poly_accuracy_layers = []
    for offset in range(1, hparams.resample_factor):
      current_grid = base_grid + 0.5 - offset / hparams.resample_factor
      method = pde.polynomials.Method.FINITE_DIFFERENCES
      poly_accuracy_layers.append(
          pde.polynomials.PolynomialAccuracyLayer(
              grid=current_grid,
              method=method,
              derivative_order=0,
              accuracy_order=hparams.polynomial_accuracy_order,
              out_scale=hparams.polynomial_accuracy_scale)
      )
    input_sizes = [layer.input_size for layer in poly_accuracy_layers]

    if hparams.num_layers > 0:
      net = pde.layers.conv1d_periodic_layer(net, filters=sum(input_sizes),
                                          kernel_size=hparams.kernel_size,
                                          activation=None, center=True)
    else:
      initializer = tf.initializers.zeros()
      coefficients = tf.get_variable(
          'coefficients', (sum(input_sizes),),
          initializer=initializer)
      net = tf.tile(coefficients[tf.newaxis, tf.newaxis, :],
                    [tf.shape(inputs)[0], inputs.shape[1].value, 1])

    cum_sizes = np.cumsum(input_sizes)
    starts = [0] + cum_sizes[:-1].tolist()
    stops = cum_sizes.tolist()
    zipped = zip(starts, stops, poly_accuracy_layers)

    outputs = tf.stack([layer.apply(net[..., start:stop])
                        for start, stop, layer in zipped], axis=-2)
    assert outputs.shape.as_list()[-1] == base_grid.size

    return outputs

def predict(inputs, hparams):
    coefficients = predict_coefficients(inputs, hparams)
    return pde.model.apply_coefficients(coefficients, inputs)

def _stack_all_rolls(inputs: tf.Tensor, max_offset: int) -> tf.Tensor:
    """Stack together all rolls of inputs, from 0 to max_offset."""
    rolled = [tf.concat([inputs[i:, ...], inputs[:i, ...]], axis=0)
                for i in range(max_offset)]
    return tf.stack(rolled, axis=0)

def stack_reconstruction(inputs, predictions):
    if isinstance(inputs, tf.Tensor):
        stacked = tf.concat([predictions, inputs[..., tf.newaxis], ], axis=-1)
        return tf.layers.flatten(stacked)
    else:
        stacked = np.concatenate([predictions, inputs[..., np.newaxis]], axis=-1)
        new_shape = stacked.shape[:-2] + (np.prod(stacked.shape[-2:]),)
        return stacked.reshape(new_shape)

def _model_inputs(fine_inputs, resample_factor):
  inputs = fine_inputs[:, resample_factor-1::resample_factor]

  labels = tf.stack([fine_inputs[:, offset-1::resample_factor]
                     for offset in range(1, resample_factor)], axis=-1)

  base_grid = pde.polynomials.regular_grid(
      pde.polynomials.GridOffset.STAGGERED, derivative_order=0,
      accuracy_order=hparams.coefficient_grid_min_size, dx=1)
  baselines = []
  for offset in range(1, hparams.resample_factor):
    current_grid = base_grid + 0.5 - offset / hparams.resample_factor
    method = pde.polynomials.Method.FINITE_DIFFERENCES
    reconstruction = pde.polynomials.reconstruct(
        inputs, current_grid, method, derivative_order=0)
    baselines.append(reconstruction)
  baseline = tf.stack(baselines, axis=-1)

  results = {'inputs': inputs, 'labels': labels, 'baseline': baseline}

  for accuracy_order in [1, 3, 5]:
    base_grid = pde.polynomials.regular_grid(
        pde.polynomials.GridOffset.STAGGERED, derivative_order=0,
        accuracy_order=accuracy_order, dx=1)
    baselines = []
    for offset in range(1, hparams.resample_factor):
      current_grid = base_grid + 0.5 - offset / hparams.resample_factor
      method = pde.polynomials.Method.FINITE_DIFFERENCES
      reconstruction = pde.polynomials.reconstruct(
          inputs, current_grid, method, derivative_order=0)
      baselines.append(reconstruction)
    results[f'baseline_{accuracy_order}'] = tf.stack(baselines, axis=-1)

  return results

def make_dataset(snapshots,
                 hparams,
                 dataset_type: Dataset = Dataset.TRAINING,
                 repeat: bool = True,
                 evaluation: bool = False) -> tf.data.Dataset:
    snapshots = np.asarray(snapshots, dtype=np.float32)

    num_training = int(round(snapshots.shape[0] * hparams.frac_training))
    if dataset_type is Dataset.TRAINING:
        indexer = slice(None, num_training)
    else:
        # assert dataset_type is Dataset.VALIDATION
        indexer = slice(num_training, None)

    dataset = tf.data.Dataset.from_tensor_slices(snapshots[indexer])
    # no need to do dataset augmentation with rolling for eval
    rolls_stop = 1 if evaluation else hparams.resample_factor
    dataset = dataset.map(lambda x: _stack_all_rolls(x, rolls_stop))
    dataset = dataset.map(lambda x: _model_inputs(x, hparams.resample_factor))
    dataset = dataset.apply(tf.data.experimental.unbatch())
    dataset = dataset.cache()

    if repeat:
        dataset = dataset.apply(
            tf.data.experimental.shuffle_and_repeat(buffer_size=10000))

    batch_size = hparams.base_batch_size * hparams.resample_factor
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    return dataset


if __name__ == '__main__':
    with h5py.File('data/8-15/1') as f:
        snapshots = f['v'][...]

    hparams = pde.training.create_hparams(
        equation='burgers',
        conservative=False,
        coefficient_grid_min_size=6,
        resample_factor=20,
        equation_kwargs=json.dumps(dict(num_points=2000)),
        base_batch_size=32,
    )


    # demonstrate a train sample
    demo_dataset = make_dataset(snapshots, hparams, repeat=False, evaluation=True)
    sess = tf.Session(config=pde.training._session_config())
    tf_example = demo_dataset.make_one_shot_iterator().get_next()
    example = sess.run(tf_example)

    plt.figure(figsize=(16, 4))
    example_id = [0, 50, 100, 150, 199]
    for id in example_id:
        plt.scatter(np.arange(0, 2000, hparams.resample_factor),
                    np.roll(example['inputs'][id], 1, axis=-1), marker='s')
        plt.plot(stack_reconstruction(example['inputs'], example['baseline_3'])[id], label='baseline')
        plt.plot(stack_reconstruction(example['inputs'], example['labels'])[id], label='exact')
        plt.legend()
        plt.show()

    # demonstrate a untrained model
    demo_dataset = make_dataset(snapshots, hparams, Dataset.VALIDATION, repeat=False, evaluation=True)

    tensors = demo_dataset.make_one_shot_iterator().get_next()
    tensors['predictions'] = predict(tensors['inputs'], hparams)
    sess.run(tf.global_variables_initializer())
    example = sess.run(tensors)

    print(example['inputs'].shape)
    for example_id in [0, 2, 4, 6, 8]:
        plt.figure(figsize=(16, 4))
        plt.scatter(np.arange(0, 2000, hparams.resample_factor),
                    np.roll(example['inputs'][example_id], 1, axis=-1), marker='s')
        plt.plot(stack_reconstruction(example['inputs'], example['baseline_3'])[example_id], label='baseline')
        plt.plot(stack_reconstruction(example['inputs'], example['labels'])[example_id], label='exact')
        plt.plot(stack_reconstruction(example['inputs'], example['predictions'])[example_id], label='predictions')
        plt.legend()
        plt.show()

    # do training
    hparams = pde.training.create_hparams(
        equation='burgers',
        conservative=True,
        coefficient_grid_min_size=6,
        resample_factor=20,
        equation_kwargs=json.dumps(dict(num_points=2000)),
        # eval_interval=500,
        base_batch_size=32,
        eval_interval=250,
        learning_stops=[1000, 2000],
        learning_rates=[3e-3, 3e-4],
    )
    loss_scale = baseline_loss(snapshots, hparams)
    tf.reset_default_graph()
    dataset = make_dataset(snapshots, hparams) # predict coefficient as in notebook
    pde.training.set_data_dependent_hparams(hparams, snapshots)
    loss, train_step = setup_training(dataset, hparams, scale=loss_scale)

    sess = tf.Session(config=pde.training._session_config())
    sess.run(tf.global_variables_initializer())

    for step in range(hparams.learning_stops[-1]):
        sess.run(train_step)
        print(step)
        if (step + 1) % hparams.eval_interval == 0:
            print(step, sess.run(loss))

    # prediction
    demo_dataset = make_dataset(snapshots, hparams, Dataset.VALIDATION, repeat=False, evaluation=True)

    tensors = demo_dataset.make_one_shot_iterator().get_next()
    tensors['predictions'] = predict(tensors['inputs'], hparams) # predict coefficient as in notebook

    array_list = []
    while True:
        try:
            array_list.append(sess.run(tensors))
        except tf.errors.OutOfRangeError:
            break

    arrays = {k: np.concatenate([d[k] for d in array_list])
            for k in array_list[0]}
            
    ds = xarray.Dataset({
        'inputs': (('sample', 'x'), arrays['inputs']),
        'labels': (('sample', 'x', 'offset'), arrays['labels']),
        'nn_predictions': (('sample', 'x', 'offset'), arrays['predictions']),
        'poly_predictions': (('sample', 'x', 'accuracy_order', 'offset'),
                            np.stack([arrays['baseline_1'],arrays['baseline_3'], arrays['baseline_5']], axis=-2)),
    }, coords={'accuracy_order': [1, 3, 5]})

    # plot hist
    plt.hist(abs(ds.labels - ds.poly_predictions.sel(accuracy_order=3)).data.ravel(),
            bins=np.geomspace(1e-6, 2, num=51), alpha=0.5, label='3rd order')
    plt.hist(abs(ds.labels - ds.nn_predictions).data.ravel(),
            bins=np.geomspace(1e-6, 2, num=51), alpha=0.5, label='neural net')
    plt.xscale('log')
    plt.legend()
    plt.show()

    # plot examples
    example_id = 0
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    x = np.arange(2000) * 2 * np.pi / 2000
    colors = seaborn.color_palette(n_colors=3)

    for ax, example_id in zip(axes.ravel(), [0, 2, 4]):
        ax.scatter(x[hparams.resample_factor-1::hparams.resample_factor],
                    ds.inputs.data[example_id],
                    marker='s',  color=colors[0])
        ax.plot(x, stack_reconstruction(ds.inputs.data, ds.labels.data)[example_id],
                label='exact', color=colors[0])
        ax.plot(x, stack_reconstruction(ds.inputs.data, ds.poly_predictions.sel(accuracy_order=3).data)[example_id],
                label='baseline', color=colors[1])
        ax.plot(x, stack_reconstruction(ds.inputs.data, ds.nn_predictions.data)[example_id],
                label='predictions', linestyle='--', color=colors[2])

    plt.show()