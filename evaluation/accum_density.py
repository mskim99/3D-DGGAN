import numpy as np

datas = []

# For Real Datas
for i in range (1, 112):
    # data = np.load('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')
    data = np.load('J:/Program/CT_VSGAN_data/gen_volume/real_data/p_' + str(i).zfill(3) + '.npy')
    data = data / 255.
    data = data[data != 0.]
    data_avg = np.average(data)

    datas.append(data_avg)
    print(data_avg)

'''
name = "230201_1_log_E_no_norm_lr_2e_6_G_D_2e_4_recon_no_IoU_cont_epoch_410_input_rate_c_n_1_1_WGAN_GP_3_RMSprop_lambda_1e_4_code_noise_random_batch_size_4"

# For Fake Datas
for i in range(0, 18):
    data = np.load('J:/Program/CT_VSGAN_data/gen_volume/' + name + '/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')

    for j in range(0, 2):
        data_portion = data[j, 0, :, :, :]
        data_portion = data_portion[data_portion != 0.]
        data_avg = np.average(data_portion)
        datas.append(data_avg)
        print(data_avg)
'''

datas = np.array(datas)
datas = datas.reshape(-1)

print(datas.shape)

np.save('J:/Program/CT_VSGAN_data/comparison/real_density_liver_avg.npy', datas)

