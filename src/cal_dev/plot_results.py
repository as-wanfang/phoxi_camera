from matplotlib import pyplot as pl
import numpy as np

data = np.load("/home/bionicdl/photoneo_data/calibration_images/data_ransac10000_valid/results_photoneo.npz")
error = data["arr_0"]
xyz = data["arr_1"]
rpy = data["arr_2"]

pl.clf()
pl.hold(1)

t = np.linspace(0, 58, 58)

r = np.mean(rpy[:,:,0], axis=0)
error = np.std(rpy[:,:,0], axis=0)
pl.plot(t, r, 'k', color='#CC4F1B', marker='o')
pl.fill_between(t, r-error, r+error,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

p = np.mean(rpy[:,:,1], axis=0)
error = np.std(rpy[:,:,1], axis=0)
pl.plot(t, p, 'k', color='#1B2ACC')
pl.fill_between(t, p-error, p+error,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

y = np.mean(rpy[:,:,2], axis=0)
error = np.std(rpy[:,:,2], axis=0)
pl.plot(t, y, 'k', color='#3F7F4C')
pl.fill_between(t, y-error, y+error,
    alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0)

pl.show()
########################################
pl.clf()
pl.hold(1)
t = np.linspace(0, 58, 58)

r = np.mean(rpy[:,:,0], axis=0)
error = np.std(rpy[:,:,0], axis=0)
pl.plot(t, r, 'k', color='#CC4F1B')
pl.fill_between(t, r-error, r+error,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

p = np.mean(rpy[:,:,1], axis=0)
error = np.std(rpy[:,:,1], axis=0)
pl.plot(t, p, 'k', color='#1B2ACC')
pl.fill_between(t, p-error, p+error,
    alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF')

y = np.mean(rpy[:,:,2], axis=0)
error = np.std(rpy[:,:,2], axis=0)
pl.plot(t, y, 'k', color='#3F7F4C')
pl.fill_between(t, y-error, y+error,
    alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99',
    linewidth=0)

pl.show()
