import numpy as np

datas = []
'''
# For Real Datas
for i in range (1, 129):
    data = np.load('J:/Program/CT_VSGAN_data/gen_volume/real_data_orig_seg_clean/p_' + str(i).zfill(3) + '.npy')
    data = data / 255.

    # data = data[data != 0.]
    data_avg = np.average(data)

    datas.append(data_avg)
    print(data_avg)

    datas.append(data)
    print(np.average(data))

datas = np.array(datas)
datas = datas.reshape(-1)

print(datas.shape)

np.save('J:/Program/CT_VSGAN_data/comparison/real_density_orig_seg_clean_avg.npy', datas)
'''

'''
s = data = np.load('J:/Program/CT_VSGAN_data/gen_volume/real_data_liver/p_001.npy')
avg_vol = np.zeros(s.shape)

for i in range (1, 112):
    data = np.load('J:/Program/CT_VSGAN_data/gen_volume/real_data_liver/p_' + str(i).zfill(3) + '.npy')
    data = data / 255.
    avg_vol = data + avg_vol
avg_vol = avg_vol / 112.

np.save('J:/Program/CT_VSGAN_data/comparison/real_density_liver_comp.npy', avg_vol)
'''

name = "230424_1_log_E_no_norm_lr_2e_6_G_ns_ref_rand_w_1_D_recon_no_IoU_cont_epoch_410_WGAN_GP_3_RMSprop_batch_size_4_orig_seg_dec_clean_smooth_250"

# For Fake Datas
for i in range(0, 32):
    # data = np.load('J:/Program/CT_VSGAN_data/gen_volume/' + name + '/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')
    # data = np.load('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230423_hagan_1e_4_4e_4_1e_4_orig_seg_num_32/hagan_volume_' + str(i) + '.npy')
    # data = np.load('J:/Program/3D-GAN-pytorch-master/imgs/230425_1_3dgan_log_lr_G_2_5e_3_D_1e_5_orig_seg_clean_num_32_ups/020000_' + str(i).zfill(2) + '.mat.npy')
    # data = np.load('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230423_hagan_1e_4_4e_4_1e_4_orig_seg_num_32/hagan_volume_' + str(i).zfill(2) + '.npy')
    data = np.load('J:/Program/3dbraingen-master/output/result/230423_1_braingen_alpha_lr_1e_4_epoch_10K_orig_result_num_32_norm/gen_volume_' + str(i).zfill(2) + '.npy')
    print(data.shape)

    for j in range(0, 1):
        # data_portion = data[j, 0, :, :, :]
        data_portion = data[:, :, :]
        '''
        # data_portion = data_portion[data_portion != 0.]
        data_avg = np.average(data_portion)
        datas.append(data_avg)
        print(data_avg)
        '''
        datas.append(data_portion)
        print(data_portion)

datas = np.array(datas)
datas = datas.reshape(-1)

print(datas.shape)

np.save('J:/Program/CT_VSGAN_data/comparison/230423_2_density_braingan_num_32.npy', datas)

