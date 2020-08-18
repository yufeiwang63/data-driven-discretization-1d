from pde_superresolution import utils  # pylint: disable=g-bad-import-order
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

input_path = 'data/8-15/2-eta'

with utils.read_h5py(input_path) as f:
    snapshots = f['v'][...]
    equation_kwargs = {k: v.item() for k, v in f.attrs.items()}

print("snapshots data: ")
print('=' * 50)
print(type(snapshots))
print(snapshots.shape)
print(snapshots)
# exit()

print("equation kwargs: ")
print("=" * 50)
print(type(equation_kwargs))
print(equation_kwargs)
# exit()

x_grid = np.linspace(0, 2, 2000)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_ylim((-1, 1.7))
line, = ax.plot(x_grid, np.zeros_like(x_grid), lw=2, label = 'solved')

def init():    
    line.set_data([], [])
    return line

def func(i):
    print('make animations, step: ', i)
    x = np.linspace(0, 2, 2000)
    y = snapshots[i]
    line.set_data(x, y)
    return line

anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=len(snapshots), interval=50)
plt.show()