import xarray
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

reference = xarray.open_dataset('./data/burgers/results.nc').isel(sample=slice(10)).load()
# print(reference)
print(reference['y'])

snapshots = reference['y'].isel(sample=0).data
print(snapshots)
print(type(snapshots))
print(snapshots.shape)
# exit()

num = snapshots.shape[1]
x_grid = np.linspace(0, 2, num)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_ylim((-1, 1.7))
line, = ax.plot(x_grid, np.zeros_like(x_grid), lw=2, label = 'solved')

def init():    
    line.set_data([], [])
    return line

def func(i):
    print('make animations, step: ', i)
    x = np.linspace(0, 2, num)
    y = snapshots[i]
    line.set_data(x, y)
    return line

anim = animation.FuncAnimation(fig=fig, func=func, init_func=init, frames=len(snapshots), interval=50)
anim.save('./data/burgers/ani.mp4', writer=writer)
plt.show()