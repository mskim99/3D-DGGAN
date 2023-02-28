from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt

df = np.load('J:/Program/CT_VSGAN_data/comparison/real_density_orig.npy', allow_pickle=True)
df = df.reshape(96, -1)
print(df.shape)

# perform multi-dimensional scaling
mds = MDS(random_state=0)
scaled_df = mds.fit_transform(df)
scaled_df = scaled_df / 1000.
# view results of multi-dimensional scaling
# print(scaled_df.shape)
# print(scaled_df)

df2 = np.load('J:/Program/CT_VSGAN_data/comparison/230223_4_density.npy', allow_pickle=True)
df2 = df2.reshape(72, -1)
scaled_df2 = mds.fit_transform(df2)
scaled_df2 = scaled_df2 / 1000.

'''
df3 = np.load('J:/Program/CT_VSGAN_data/comparison/230223_2_density.npy', allow_pickle=True)
df3 = df3.reshape(36, -1)
scaled_df3 = mds.fit_transform(df3)
scaled_df3 = scaled_df3 / 1000.

df4 = np.load('J:/Program/CT_VSGAN_data/comparison/230223_3_density.npy', allow_pickle=True)
df4 = df4.reshape(36, -1)
scaled_df4 = mds.fit_transform(df4)
scaled_df4 = scaled_df3 / 1000.
'''

fig = plt.figure()
ax1 = fig.add_subplot(111)

# create scatter plot
ax1.scatter(scaled_df[:,0], scaled_df[:,1], color='blue', s=20, marker="s", label='real')
ax1.scatter(scaled_df2[:,0], scaled_df2[:,1], color='red', s=20, marker="o", label='230223_4')
# ax1.scatter(scaled_df3[:,0], scaled_df3[:,1], color='orange', s=20, marker="o", label='230223_2')
# ax1.scatter(scaled_df4[:,0], scaled_df4[:,1], color='yellow', s=20, marker="o", label='230223_3')
plt.legend(loc='upper left')

# display scatterplot
plt.show()