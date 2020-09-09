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
import os
import torch
from matplotlib import animation
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


# use the model to do integration
class ModelDifferentiator(pde.integrate.Differentiator):
    """Calculate derivatives from a current TensorFlow Model"""
    def __init__(self,
                num_points,
                hparams: tf.contrib.training.HParams,
                model_path):

        with tf.Graph().as_default():
            self.t = tf.placeholder(tf.float32, shape=())

            num_points = num_points
            self.inputs = tf.placeholder(tf.float32, shape=(num_points,))

            time_derivative = tf.squeeze(model.predict_time_derivative(
                self.inputs[tf.newaxis, :], hparams), axis=0)
            self.value = time_derivative # directly ignore forcing here

            saver = tf.train.Saver()
            self.sess = tf.Session()
            saver.restore(self.sess, model_path)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        return self.sess.run(self.value, feed_dict={self.t: t, self.inputs: y})


def make_animation(save_path=None, save_name=None, precise_solution=None, args=None):
    x_grid = solution_x
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_ylim((np.min(solution_model[0]), np.max(solution_model[0])))
    line, = ax.plot(x_grid, np.zeros_like(x_grid), marker='o', lw=2, label = 'l3d')

    precise_dx = args.precise_dx
    precise_num = int(2 / precise_dx) + 1
    precise_x_grid = np.linspace(0, 2, precise_num)
    line_precise, = ax.plot(precise_x_grid, np.zeros_like(precise_x_grid), lw=2, label = 'reference')
    

    def init():    
        line.set_data([], [])
        line_precise.set_data([], [])
        return line

    def func(i):
        # print('make animations, step: ', i)
        x = np.linspace(0, 2, num)
        y = solution_model[i]
        line.set_data(x, y)

        x = np.linspace(0, 2, precise_num)
        precise_t_idx = int(i * args.dx / args.precise_dx)
        y = precise_solution[precise_t_idx]
        line_precise.set_data(x, y)
        return line

    anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=len(solution_model), interval=50)
    if save_path is not None:
        anim.save(os.path.join(save_path, save_name), writer=writer)
    else:
        plt.show()

def compute_error(corase, precise, dx, precise_dx, dt, precise_dt):
    
    errors = []
    for idx in range(len(corase)):
        t = dt * idx
        precise_t_idx = t / precise_dt
        precise_val = [precise_t_idx]
        factor = int(dx / precise_dx)
        precise_val = precise_val[::factor]
        error_t = np.linalg.norm(precise_val - corase[idx], 2) / np.linalg.norm(precise_val, 2)    
        errors.append(error_t)

    return np.mean(errors)


model_path = 'data/burgers-checkpoints-paper-data/model.ckpt-40000'
equation_kwargs = {}
equation_kwargs['num_points'] = 2000
equation_kwargs['eta'] = 0.0
equation_kwargs['k_min'] = 3
equation_kwargs['k_max'] = 6
precise_dx = 0.001

dt_list = [0.02, 0.04, 0.05]
dx_list = [0.02, 0.04, 0.05]
l3d_mean_errors = np.zeros((len(dt_list), len(dx_list)))
l3d_all_errors = []

for eta in [0]:
    if eta == 0:
        init_data_path = '/media/yufei/drive2/yufei_data/RLPDE-new/RLPDE-v4/data/local/solutions/8-14-50'
    else:
        init_data_path = '/media/yufei/drive2/yufei_data/RLPDE-new/RLPDE-v4/data/local/solutions/9-8-50-eta-{}'.format(eta)

    all_files_unfiltered = os.listdir(init_data_path)
    all_files = []
    for x in all_files_unfiltered:
        if '.pkl' in x:
            all_files.append(x)
    all_files.sort()

    for t_idx, dt in enumerate(dt_list):
        for x_idx, dx in enumerate(dx_list):
            resample_factor = int(dx / precise_dx)

            equation_kwargs['eta'] = eta
            hparams = pde.training.create_hparams(
                equation='burgers',
                conservative=True,
                resample_factor=resample_factor,
                equation_kwargs=json.dumps(equation_kwargs)
            )

            errors = []
            for idx in range(len(all_files) // 2, len(all_files)):
                init_file = all_files[idx]
                print(init_file)
                data = torch.load(os.path.join(init_data_path, init_file))
                a, b, c, d, e = data['a'], data['b'], data['c'], data['d'], data['e']
                precise_solution = data['precise_solution']
                args = data['args']
                dt = args.dx * args.cfl
                T = args.T

                num = equation_kwargs['num_points'] // resample_factor # grid point num
                solution_x = np.linspace(0, 2, num) # solution period
                y0 = a + b * np.sin(c * np.pi * solution_x) + d * np.cos(e * np.pi * solution_x)
                times = np.arange(0, T, dt) # integration time
                differentiator = ModelDifferentiator(num, hparams, model_path)
                solution_model, num_evals_model = pde.integrate.odeint(
                    y0, differentiator, times, method='RK23')

                error = compute_error(solution_model, precise_solution, 2 / 31, args.precise_dx, dt, args.precise_dx * args.cfl)
                print(f"init {init_file} error {error}")

                errors.append(error)

                # make_animation(save_path='data/notion_demonstration', 
                #     save_name='{}-{}-{}.mp4'.format(idx, equation_kwargs['eta'], resample_factor),
                #     precise_solution=precise_solution,
                #     args=args)
                # make_animation(save_path='data/notion_demonstration', 
                #     save_name='{}-{}-{}.mp4'.format(idx, equation_kwargs['eta'], resample_factor),
                #     precise_solution=precise_solution,
                #     args=args)

            l3d_all_errors.append(errors)
            l3d_mean_errors[t_idx][x_idx] = np.mean(errors)

data_dict = {
    'dt_list': dt_list,
    'dx_list': dx_list,
    'all_errors_l3d': l3d_all_errors,
    'mean_errors_l3d': l3d_mean_errors,
}

torch.save(data_dict, 'data/ComPhy/l3d_error_{}.pkl'.format(eta))

# for t_idx, dt in enumerate(dt_list):
#     print(dt, end=' ')
#     for x_idx, dx in enumerate(dx_list):
#         l3d_error = round(l3d_mean_errors[t_idx][x_idx] * 100, 2)
#         print("\& {} \& {}".format(l3d_error, weno_error), end=' ')
#     print('\\\hline')
