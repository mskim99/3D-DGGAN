import numpy as np
import binvox_rw
import os

'''
data = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/220910_2_log_loss_GAN_L1_IoU_loss/epoch_200_real_B_07.npy')
volume_path = 'J:/Program/vox2vox-master/vox2vox-master/data/test/volume/f_0000009/model.binvox'
with open(volume_path, 'rb') as f:
    volume = binvox_rw.read_as_3d_array(f)
data = volume.data
print(data.min())
print(data.max())
'''
'''
vol_num = 0
avg_vol = np.zeros([128, 128, 128])

# Generate Averaged volume
for i in range (1, 53):
    volume_path = './data/orig/train/volume/f_' + str(i).zfill(7) + '/model.binvox'
    if os.path.exists(volume_path):
        with open(volume_path, 'rb') as f:
            volume = binvox_rw.read_as_3d_array(f)
        data = volume.data

        avg_vol = avg_vol + data
        vol_num = vol_num + 1

for i in range (1, 52):
    volume_path = './data/orig/train/volume/m_' + str(i).zfill(7) + '/model.binvox'
    if os.path.exists(volume_path):
        with open(volume_path, 'rb') as f:
            volume = binvox_rw.read_as_3d_array(f)
        data = volume.data

        avg_vol = avg_vol + data
        vol_num = vol_num + 1

avg_vol = avg_vol / float(vol_num)

voxels = binvox_rw.from_array(avg_vol, [128, 128, 128], [0.0, 0.0, 0.0], 1, fix_coords=True)
with open('./data/orig/train/volume/avg_volume.binvox', 'wb') as f:
    voxels.write(f)

print(vol_num)
print('Finished')
'''

datas = []

for i in range (0, 18):
    data = np.load('J:/Program/CT_VSGAN/gen_volume/230113_2_log_E_lr_2e_6_G_D_recon_no_IoU_cont_epoch_410_input_rate_c_n_1_1_WGAN_GP_RMSprop_lambda_1e_3/epoch_400_fake_EC_' + str(i).zfill(2) + '.npy')
    print(np.average(data))
    print(np.std(data))
    datas.append(data)

datas = np.array(datas)

for i in range(0, 18):
    print(np.abs(datas[i] - datas[0]).sum() / 262144.)
