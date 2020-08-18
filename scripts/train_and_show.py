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
from pde_superresolution.model import Dataset

def setup_training(dataset, hparams, scale=1.0):
    # predict u, u_x, u_t as in training
    tensors = dataset.make_one_shot_iterator().get_next()

    predictions = model.predict_result(tensors['inputs'], hparams)

    loss_per_head = model.loss_per_head(predictions,
                                        labels=tensors['labels'],
                                        baseline=tensors['baseline'],
                                        hparams=hparams)
    loss = model.weighted_loss(loss_per_head, hparams)
    train_step = pde.training.create_training_step(loss, hparams)
    return loss, train_step


with h5py.File('data/8-15/1') as f:
    snapshots = f['v'][...]

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
    learning_stops=[50, 100], # traiining time
    learning_rates=[3e-3, 3e-4],
)
tf.reset_default_graph()
dataset = pde.model.make_dataset(snapshots, hparams) # predict u, u_x, u_t as in training
pde.training.set_data_dependent_hparams(hparams, snapshots)
loss, train_step = setup_training(dataset, hparams)

sess = tf.Session(config=pde.training._session_config())
sess.run(tf.global_variables_initializer())

for step in range(hparams.learning_stops[-1]):
    sess.run(train_step)
    print(step)
    if (step + 1) % hparams.eval_interval == 0:
        print(step, sess.run(loss))

# save
save_path = 'tmp_model.ckpt'
saver = tf.train.Saver()
saver.save(sess, save_path)

# prediction
demo_dataset = pde.model.make_dataset(snapshots, hparams, Dataset.VALIDATION, repeat=False, evaluation=True)
tensors = demo_dataset.make_one_shot_iterator().get_next()
tensors['predictions'] = model.predict_result(tensors['inputs'], hparams)

array_list = []
while True:
    try:
        array_list.append(sess.run(tensors))
    except tf.errors.OutOfRangeError:
        break

arrays = {k: np.concatenate([d[k] for d in array_list])
          for k in array_list[0]}

# do some simple plots
# inputs = arrays['inputs']
# predictions = arrays['predictions'][:, :, 0]
# labels = arrays['labels'][:, :, 0]
# print("inputs shape: ", inputs.shape)
# print("predictions shape: ", predictions.shape)
# print("labels shape: ", labels.shape)
# example_idx = [0, 10, 20, 30, 40]
# for idx in example_idx:
#     true = labels[idx]
#     predict = predictions[idx]
#     plt.plot(range(len(true)), true, label='true')
#     plt.plot(range(len(predict)), predict, label='predict')
#     plt.show()


# use the model to do integration
class ModelDifferentiator(pde.integrate.Differentiator):
    """Calculate derivatives from a current TensorFlow Model"""
    def __init__(self,
                num_points,
                hparams: tf.contrib.training.HParams):

        with tf.Graph().as_default():
            self.t = tf.placeholder(tf.float32, shape=())

            num_points = num_points
            self.inputs = tf.placeholder(tf.float32, shape=(num_points,))

            time_derivative = tf.squeeze(model.predict_time_derivative(
                self.inputs[tf.newaxis, :], hparams), axis=0)
            self.value = time_derivative

            saver = tf.train.Saver()
            self.sess = tf.Session()
            saver.restore(self.sess, save_path)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.sess.run(self.value, feed_dict={self.t: t, self.inputs: y})

num = 100 # grid point num
solution_x = np.linspace(0, 2, num) # solution period
y0 = 0.5 + np.sin(2 * np.pi * solution_x)
times = np.arange(0, 0.5, 0.002) # integration time
differentiator = ModelDifferentiator(num, hparams)
solution_model, num_evals_model = pde.integrate.odeint(
    y0, differentiator, times, method='RK23')

# make animations
from matplotlib import animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

x_grid = solution_x
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_ylim((-1, 1.7))
line, = ax.plot(x_grid, np.zeros_like(x_grid), lw=2, label = 'solved')

def init():    
    line.set_data([], [])
    return line

def func(i):
    print('make animations, step: ', i)
    x = np.linspace(0, 2, num)
    y = solution_model[i]
    line.set_data(x, y)
    return line

anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=len(solution_model), interval=50)
# anim.save('./data/burgers/ani.mp4', writer=writer)
plt.show()