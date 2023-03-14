import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style Initialization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['font.size'] = 9

# Data Preparation
np.random.seed(0)
'''
data_a = np.random.normal(0.5, 0.125, 500)
data_b = np.random.normal(0.5, 0.125, 1000)
data_c = np.random.normal(0.5, 0.125, 1500)
'''

data_a = np.load('J:/Program/CT_VSGAN_data/comparison/230130_5_density_avg.npy')
print('data_a finished')
data_b = np.load('J:/Program/CT_VSGAN_data/comparison/230201_1_density_avg.npy')
print('data_b finished')
data_c = np.load('J:/Program/CT_VSGAN_data/comparison/real_density_liver_avg.npy')
print('data_c finished')

# Draw Graph
fig, ax = plt.subplots()
bp = ax.boxplot([data_a, data_b, data_c], notch='True')

colors = ['cyan', 'lightblue', 'lightgreen']
for artist, color in zip(bp['boxes'], colors):
    patch = mpatches.PathPatch(artist.get_path(), color=color)
    ax.add_artist(patch)

ax.yaxis.grid(True)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Label')
ax.set_ylabel('Value')
plt.xticks([1, 2, 3], ['230130_5', '230201_1', 'real'])

plt.show()