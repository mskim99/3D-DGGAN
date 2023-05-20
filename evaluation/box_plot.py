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
data_a = np.load('J:/Program/CT_VSGAN_data/comparison/230425_1_density_3dgan_num_32.npy')
data_a = data_a[data_a != 0]
print('data_a finished')
data_b = np.load('J:/Program/CT_VSGAN_data/comparison/230423_1_density_hagan_num_32.npy')
data_b = data_b[data_b < 0.31]
print('data_b finished')
data_c = np.load('J:/Program/CT_VSGAN_data/comparison/230423_2_density_braingan_num_32.npy')
data_c = data_c[data_c != 0]
print('data_c finished')
data_d = np.load('J:/Program/CT_VSGAN_data/comparison/230424_1_density_num_32.npy')
data_d = data_d[data_d != 0]
print('data_d finished')
data_e = np.load('J:/Program/CT_VSGAN_data/comparison/real_density_orig_seg_clean.npy')
data_e = data_e[data_e != 0]
print('data_e finished')

# Draw Graph
fig, ax = plt.subplots()
bp = ax.boxplot([data_a, data_b, data_c, data_d, data_e], notch='True', showfliers=False)

colors = ['orange', 'yellow', 'green', 'red', 'blue']
for artist, color in zip(bp['boxes'], colors):
    patch = mpatches.PathPatch(artist.get_path(), color=color)
    ax.add_artist(patch)

ax.yaxis.grid(True)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel('Label')
ax.set_ylabel('Value')
plt.title('Spine')
plt.xticks([1, 2, 3, 4, 5], ['3D-GAN', 'HA-GAN', '3D-α-GAN', 'Proposed', 'real'])

plt.show()