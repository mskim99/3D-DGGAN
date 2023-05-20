import torch
import numpy as np
import piq
import os

def msssim_volume(gv, gtv, normalize=True):

    loss_total = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, i]).cpu().numpy()
        gtv_part = (gtv[:, :, :, i]).cpu().numpy()

        gv_process = np.where(gtv_part > 0, gv_part, 0)
        gtv_part = torch.tensor(gtv_part).cuda()
        gv_process = torch.tensor(gv_process).cuda()

        gv_process = gv_process.reshape(1, 3, 128, 128)
        gtv_part = gtv_part.reshape(1, 3, 128, 128)

        loss_part = piq.multi_scale_ssim(gv_process, gtv_part, kernel_size=7)
        loss_total += loss_part

    if normalize:
        loss_total = loss_total / gv.shape[2]

    return loss_total


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

device = "cuda" if torch.cuda.is_available() else "cpu"

res_batch_num = 1
MS_SSIM = 0.

volumes = []

for i in range (1, 112):
    # fake_volume = np.load('../comparison/230424_1_log_E_no_norm_lr_2e_6_G_ns_ref_rand_w_1_D_recon_no_IoU_cont_epoch_410_WGAN_GP_3_RMSprop_batch_size_4_orig_seg_dec_clean_smooth_250/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')
    fake_volume = np.load('../comparison/real_data_liver/p_' + str(i).zfill(3) + '.npy')
    # fake_volume = np.load('../comparison/230423_hagan_1e_4_4e_4_1e_4_orig_seg/hagan_volume_' + str(i).zfill(2) + '.npy')
    # fake_volume = np.load('../comparison/230425_1_3dgan_log_lr_G_2_5e_3_D_1e_5_orig_seg_clean_ups/020000_' + str(i).zfill(2) + '.mat.npy')
    # fake_volume = np.load('../comparison/230423_1_braingen_alpha_lr_1e_4_epoch_6K_orig_result_norm/gen_volume_' + str(i).zfill(2) + '.npy')
    # fake_volume = np.load('../comparison/230204_liver_vox2vox/epoch_400_real_B_' + str(i).zfill(2) + '.npy')

    fake_volume = fake_volume / 255.
    fake_volume = torch.from_numpy(fake_volume)
    # print(fake_volume.shape)

    for kf in range (0, res_batch_num):

        # fake_volume_part = fake_volume[kf, :, :, :, :]
        fake_volume_part = fake_volume[:, :, :]

        fake_volume_part = fake_volume_part.squeeze()
        fake_volume_part = fake_volume_part.unsqueeze(1)
        fake_volume_part = torch.concat([fake_volume_part, fake_volume_part, fake_volume_part], dim=1)
        fake_volume_part = (fake_volume_part).to(device)
        volumes.append(fake_volume_part)

for i in range (0, 111):
    for j in range(0, 111):
        if i < j:
            volume1 = volumes[i]
            volume2 = volumes[j]
            MS_SSIM = MS_SSIM + msssim_volume(volume1, volume2).item()
            print(str(i) + ' / ' + str(j) + ' MS-SSIM: ' + str(MS_SSIM))

MS_SSIM = MS_SSIM / (56. * 111.)
print('Final MS-SSIM: ' + str(MS_SSIM))