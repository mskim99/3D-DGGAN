import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import skimage.metrics as ski_metrics
import piq
import inception_score as inceptionScore
import os
import numpy as np
import math


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

        gtv_part.detach()
        gv_process.detach()

    return loss_total


def psnr_volume(gv, gtv, normalize=True):

    psnr_value_total = 0.0
    psnr_value_valid = 0
    for i in range (0, gv.shape[2]):
        gv_part = (gv[:, :, :, i]).cpu().numpy()
        gtv_part = (gtv[:, :, :, i]).cpu().numpy()

        psnr_value_part = ski_metrics.peak_signal_noise_ratio(gv_part, gtv_part)
        if not np.isinf(psnr_value_part):
            psnr_value_total += psnr_value_part
            psnr_value_valid = psnr_value_valid + 1

    if normalize and psnr_value_valid > 0:
        psnr_value_total = psnr_value_total / psnr_value_valid

    return psnr_value_total


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

device = "cuda" if torch.cuda.is_available() else "cpu"

fid = FrechetInceptionDistance(feature=768).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
kid = KernelInceptionDistance(feature=768, subset_size=50).to(device)

# generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 200, (128, 3, 128, 128), dtype=torch.uint8)
# imgs_dist2 = torch.randint(100, 255, (128, 3, 128, 128), dtype=torch.uint8)

res_batch_num = 4
KID = 0.
FID = 0.
LPIPS = 0.
IS = 0.
MS_SSIM = 0.
PSNR = 0.

for i in range (0, 4):
    fake_volume = np.load('../comparison/230505_1_log_E_no_norm_lr_2e_6_G_ns_ref_rand_w_1_D_sgan_slab_18_recon_no_IoU_cont_epoch_410_WGAN_GP_3_RMSprop_batch_size_4_orig_seg_clean/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')
    # fake_volume = np.load('../comparison/230423_hagan_1e_4_4e_4_1e_4_orig_seg_num_32/hagan_volume_' + str(i).zfill(2) + '.npy')
    # fake_volume = np.load('../comparison/230425_1_3dgan_log_lr_G_2_5e_3_D_1e_5_orig_seg_clean_num_32_ups/020000_' + str(i).zfill(2) + '.mat.npy')
    # fake_volume = np.load('../comparison/230423_1_braingen_alpha_lr_1e_4_epoch_10K_orig_result_num_32_norm/gen_volume_' + str(i).zfill(2) + '.npy')
    # fake_volume = np.load('../comparison/230204_liver_vox2vox/epoch_400_real_B_' + str(i).zfill(2) + '.npy')

    fake_volume = torch.from_numpy(fake_volume)
    '''
    fake_volume = fake_volume.squeeze()
    fake_volume = fake_volume.unsqueeze(1)
    fake_volume = torch.concat([fake_volume, fake_volume, fake_volume], dim=1)
    fake_volume_int = (fake_volume * 255.).type(torch.uint8).to(device)
    fake_volume = (fake_volume - 1.).to(device)
    '''

    # print(fake_volume.shape)

    for kf in range (0, res_batch_num):

        fake_volume_part = fake_volume[kf, :, :, :, :]
        # fake_volume_part = fake_volume[:, :, :]

        fake_volume_part = fake_volume_part.squeeze()
        fake_volume_part = fake_volume_part.unsqueeze(1)
        fake_volume_part = torch.concat([fake_volume_part, fake_volume_part, fake_volume_part], dim=1)
        fake_volume_int = (fake_volume_part * 255.).type(torch.uint8).to(device)
        fake_volume_part = (fake_volume_part).to(device)

        # print(fake_volume_part.shape)

        IS = IS + inceptionScore.inception_score(fake_volume_part, cuda=True, batch_size=18, resize=True, splits=1)[0]
        print(str(i) + '_' + str(kf) + ' : Finished, IS : ' + str(IS))

        for j in range (1, 128):
            real_volume = np.load('../comparison/real_data_orig_seg_clean/p_' + str(j).zfill(3) + '.npy')

            real_volume_part = real_volume

            # print(real_volume_part.min())
            # print(real_volume_part.max())

            real_volume_part = torch.from_numpy(real_volume_part)
            real_volume_part = real_volume_part.squeeze()
            real_volume_part = real_volume_part.unsqueeze(1)
            real_volume_part = torch.concat([real_volume_part, real_volume_part, real_volume_part], dim=1)
            real_volume_int = (real_volume_part).type(torch.uint8).to(device)
            real_volume_part = (real_volume_part / 255.).to(device)

            # print(real_volume_int.shape)
            # print(fake_volume_int.shape)

            fid.update(real_volume_int, real=True)
            fid.update(fake_volume_int, real=False)

            kid.update(real_volume_int, real=True)
            kid.update(fake_volume_int, real=False)

            KID_mean, _ = kid.compute()
            KID = KID + KID_mean
            FID = FID + fid.compute()
            LPIPS = LPIPS + lpips(real_volume_part, fake_volume_part).detach()
            MS_SSIM = MS_SSIM + msssim_volume(real_volume_part, fake_volume_part).item()
            PSNR = PSNR + psnr_volume(real_volume_part, fake_volume_part).item()

            print(str(i) + '_' + str(kf) + ' / ' + str(j) + ' : Finished' +
                ', KID : ' + str(KID) + ', FID : ' + str(FID) + ', LPIPS : ' + str(LPIPS) +
                ', MS_SSIM : ' + str(MS_SSIM) + '. PSNR : ' + str(PSNR))

            fake_volume_part.detach()
            fake_volume_int.detach()

        fake_volume_part.detach()
        fake_volume_int.detach()

    '''
    real_volume = np.load('../comparison/real_density_liver_comp.npy')

    real_volume_part = real_volume

    real_volume_part = torch.from_numpy(real_volume_part)
    real_volume_part = real_volume_part.squeeze()
    real_volume_part = real_volume_part.unsqueeze(1)
    real_volume_part = torch.concat([real_volume_part, real_volume_part, real_volume_part], dim=1)
    real_volume_int = (real_volume_part).type(torch.uint8).to(device)
    real_volume_part = (real_volume_part / 255.).type(torch.float).to(device)

    fid.update(real_volume_int, real=True)
    fid.update(fake_volume_int, real=False)

    kid.update(real_volume_int, real=True)
    kid.update(fake_volume_int, real=False)

    KID_mean, _ = kid.compute()
    KID = KID + KID_mean
    FID = FID + fid.compute()
    LPIPS = LPIPS + lpips(real_volume_part, fake_volume_part).detach()
    MS_SSIM = MS_SSIM + msssim_volume(real_volume_part, fake_volume_part).item()
    PSNR = PSNR + psnr_volume(real_volume_part, fake_volume_part).item()

    print(str(i) + '_' + str(kf) + ' : Finished' +
          ', KID : ' + str(KID) + ', FID : ' + str(FID) + ', LPIPS : ' + str(LPIPS) +
          ', MS_SSIM : ' + str(MS_SSIM) + '. PSNR : ' + str(PSNR))
            '''

IS = IS / (16.)
KID = KID / (16. * 128.)
MMD = torch.sqrt(KID)
FID = FID / (16. * 128.)
LPIPS = LPIPS / (16. * 128.)
MS_SSIM = MS_SSIM / (16. * 128.)
PSNR = PSNR / (16. * 128.)

print('IS : ' + str(IS))
print('KID : ' + str(KID.item()))
print('MMD : ' + str(MMD.item()))
print('FID : ' + str(FID.item()))
print('LPIPS : ' + str(LPIPS.item()))
print('MS_SSIM : ' + str(MS_SSIM))
print('PSNR : ' + str(PSNR))