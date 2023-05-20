from scipy.spatial import ConvexHull
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt

df = np.load('J:/Program/CT_VSGAN_data/comparison/real_density_orig_seg_clean.npy', allow_pickle=True)
df = df.reshape(128, -1)
# print(df.shape)

# perform multi-dimensional scaling
# mds = MDS(random_state=0)
mds = MDS(random_state=0)
scaled_df = mds.fit_transform(df)
scaled_df = scaled_df / 1000.
hull_df = ConvexHull(scaled_df)

# print(scaled_df.shape)
# view results of multi-dimensional scaling
# print(scaled_df.shape)
# print(scaled_df)

df2 = np.load('J:/Program/CT_VSGAN_data/comparison/230425_1_density_3dgan_num_32.npy', allow_pickle=True)
df2 = df2.reshape(128, -1)
scaled_df2 = mds.fit_transform(df2)
scaled_df2 = scaled_df2 / 1000.
hull_df2 = ConvexHull(scaled_df2)

df3 = np.load('J:/Program/CT_VSGAN_data/comparison/230423_1_density_hagan_num_32.npy', allow_pickle=True)
df3 = df3.reshape(128, -1)
scaled_df3 = mds.fit_transform(df3)
scaled_df3 = scaled_df3 / 1000.
hull_df3 = ConvexHull(scaled_df3)

df4 = np.load('J:/Program/CT_VSGAN_data/comparison/230423_2_density_braingan_num_32.npy', allow_pickle=True)
df4 = df4.reshape(128, -1)
scaled_df4 = mds.fit_transform(df4)
scaled_df4 = scaled_df4 / 1000.
hull_df4 = ConvexHull(scaled_df4)

df5 = np.load('J:/Program/CT_VSGAN_data/comparison/230424_1_density_num_32.npy', allow_pickle=True)
df5 = df5.reshape(128, -1)
scaled_df5 = mds.fit_transform(df5)
scaled_df5 = scaled_df5 / 1000.
hull_df5 = ConvexHull(scaled_df5)


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(111)
# ax1 = fig.gca(projection='3d')

# create scatter plot (2D)
ax1.scatter(scaled_df[:,0], scaled_df[:,1], color='blue', s=20, marker="s", label='real')
ax1.scatter(scaled_df2[:,0], scaled_df2[:,1], color='orange', s=20, marker="o", label='3D-GAN')
ax1.scatter(scaled_df3[:,0], scaled_df3[:,1], color='yellow', s=20, marker="o", label='HA-GAN')
ax1.scatter(scaled_df4[:,0], scaled_df4[:,1], color='green', s=20, marker="o", label='3D-α-GAN')
ax1.scatter(scaled_df5[:,0], scaled_df5[:,1], color='red', s=20, marker="o", label='Proposed')

for simplex in hull_df.simplices:
    ax1.plot(scaled_df[simplex, 0], scaled_df[simplex, 1], color='blue')
for simplex in hull_df2.simplices:
    ax1.plot(scaled_df2[simplex, 0], scaled_df2[simplex, 1], color='orange')
for simplex in hull_df3.simplices:
    ax1.plot(scaled_df3[simplex, 0], scaled_df3[simplex, 1], color='yellow')
for simplex in hull_df4.simplices:
    ax1.plot(scaled_df4[simplex, 0], scaled_df4[simplex, 1], color='green')
for simplex in hull_df5.simplices:
    ax1.plot(scaled_df5[simplex, 0], scaled_df5[simplex, 1], color='red')
# ax1.plot(scaled_df[hull_df.vertices, 0], scaled_df[hull_df.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)

# create scatter plot (3D)
'''
ax1.scatter(scaled_df[:,0], scaled_df[:,1], scaled_df[:,2], color='blue', s=20, marker="s", label='real')
ax1.scatter(scaled_df2[:,0], scaled_df2[:,1], scaled_df2[:,2], color='orange', s=20, marker="o", label='3D-GAN')
ax1.scatter(scaled_df3[:,0], scaled_df3[:,1], scaled_df3[:,2], color='yellow', s=20, marker="o", label='HA-GAN')
ax1.scatter(scaled_df4[:,0], scaled_df4[:,1], scaled_df4[:,2], color='green', s=20, marker="o", label='Brain-GAN')
ax1.scatter(scaled_df5[:,0], scaled_df5[:,1], scaled_df5[:,2], color='red', s=20, marker="o", label='Proposed')
'''

plt.legend(loc='upper left')
plt.title('spine', pad=10, fontsize=20)

# display scatterplot
plt.show()