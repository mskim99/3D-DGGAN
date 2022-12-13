import torch

import pytorch_fid.fid_score as fs
import numpy as np

from ignite.metrics.gan import FID
import inception_score as IS


fake_volumes = []
real_volumes = []

for i in range (0, 18):
    fake_volume = np.load('./comparison/221213_2_log_loss_vol_slab_slice_G_recon_D_5_sigmoid_clamp_epoch_610/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')
    real_volume = np.load('./comparison/real_data/real_V_' + str(i).zfill(2) + '.npy')
    fake_volumes.append(fake_volume)
    real_volumes.append(real_volume)

fake_volumes = torch.Tensor(fake_volumes)
fake_volumes = fake_volumes.squeeze(1)
fake_volumes = fake_volumes.squeeze(1)

fake_volumes = fake_volumes.reshape(-1, 1, 128, 128)


real_volumes = torch.Tensor(real_volumes)
real_volumes = fake_volumes.squeeze(1)
real_volumes = fake_volumes.squeeze(1)

real_volumes = real_volumes.reshape(-1, 1, 128, 128)


fake_volumes = torch.concat([fake_volumes, fake_volumes, fake_volumes], dim=1)
real_volumes = torch.concat([real_volumes, real_volumes, real_volumes], dim=1)


IS_score = IS.inception_score(real_volumes, cuda=True, batch_size=18, resize=True, splits=1)
print(IS_score)


# fake_volumes = fake_volumes.reshape(-1, 128, 128, 3)
# real_volumes = real_volumes.reshape(-1, 128, 128, 3)

'''
fake_volumes = fake_volumes * 255.
real_volumes = real_volumes * 255.
fake_volumes = fake_volumes.astype(np.uint8)
real_volumes = real_volumes.astype(np.uint8)
'''

FID_metric = FID()
# for i in range(0, fake_volumes.shape[0]):
FID_metric.update((fake_volumes, real_volumes))
print(FID_metric.compute())