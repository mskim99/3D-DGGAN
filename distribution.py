import numpy as np

total_vol_num = 0
total_vol_num2 = 0
vox_num = 128 * 128 * 128

for i in range (0, 18):
    real_B = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/221005_3_log_loss_GAN_vox2vox_normalize_L2_uqi_loss_L1_33_layer_4_LFFB/epoch_200_real_B_' + str(i).zfill(2) + '.npy')
    real_B2 = np.load('J:/Program/vox2vox-master/vox2vox-master/logs/221019_1_log_loss_GAN_vox2vox_normalize_L2_lol2_uqi_loss_layer_4_LFFB_rm/epoch_200_real_B_' + str(i).zfill(2) + '.npy')

    vol_num = np.greater(real_B, 0.4).sum()
    vol_num2 = np.greater(real_B2, 0.4).sum()

    total_vol_num = total_vol_num + vol_num
    total_vol_num2 = total_vol_num2 + vol_num2

print(float(total_vol_num) / float(vox_num))
print(float(total_vol_num2) / float(vox_num))