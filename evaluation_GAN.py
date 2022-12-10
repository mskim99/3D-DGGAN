import torch

import pytorch_fid.fid_score as fs
import numpy as np

import inception_score as IS

fake_volumes = []
# real_volumes = []

for i in range (0, 18):
    fake_volume = np.load('./comparison/221209_4_log_loss_vol_slab_slice_G_recon_sigmoid_clamp_epoch_410_lr_2e-6/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')
    # real_volume = np.load('./comparison/epoch_200_real_V_' + str(i).zfill(2) + '.npy')
    fake_volumes.append(fake_volume)
    # real_volumes.append(real_volume)

fake_volumes = torch.Tensor(fake_volumes)
fake_volumes = fake_volumes.squeeze(1)
fake_volumes = fake_volumes.squeeze(1)

fake_volumes = fake_volumes.reshape(-1, 1, 128, 128)

fake_volumes = torch.concat([fake_volumes, fake_volumes, fake_volumes], dim=1)

IS_score = IS.inception_score(fake_volumes, cuda=True, batch_size=18, resize=True, splits=1)

print(IS_score)

'''
real_volumes = torch.Tensor(real_volumes)
real_volumes = real_volumes.squeeze(1)
real_volumes = real_volumes.squeeze(1)
'''
