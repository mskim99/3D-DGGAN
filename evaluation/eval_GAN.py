import torch
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import inception_score as inceptionScore
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

device = "cuda" if torch.cuda.is_available() else "cpu"

fid = FrechetInceptionDistance(feature=64).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
kid = KernelInceptionDistance(subset_size=50).to(device)

# generate two slightly overlapping image intensity distributions
# imgs_dist1 = torch.randint(0, 200, (128, 3, 128, 128), dtype=torch.uint8)
# imgs_dist2 = torch.randint(100, 255, (128, 3, 128, 128), dtype=torch.uint8)

res_batch_num = 2

KID = 0.
FID = 0.
LPIPS = 0.
IS = 0.
for i in range (0, 18):
    fake_volume = np.load('../comparison/230130_4_log_E_no_norm_lr_2e_6_G_D_recon_no_IoU_cont_epoch_410_input_rate_c_n_1_1_WGAN_GP_1_RMSprop_lambda_1e_4_code_noise_random_batch_size_2/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')

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

        fake_volume_part = fake_volume_part.squeeze()
        fake_volume_part = fake_volume_part.unsqueeze(1)
        fake_volume_part = torch.concat([fake_volume_part, fake_volume_part, fake_volume_part], dim=1)
        fake_volume_int = (fake_volume_part * 255.).type(torch.uint8).to(device)
        fake_volume_part = (fake_volume_part - 1.).to(device)

        # print(fake_volume_part.shape)

        IS = IS + inceptionScore.inception_score(fake_volume_part, cuda=True, batch_size=18, resize=True, splits=1)[0]
        print(str(i) + '_' + str(kf) + ' : Finished, IS : ' + str(IS))

        for j in range (0, 18):
            real_volume = np.load('../comparison/real_data/real_V_' + str(j).zfill(2) + '.npy')

            for kr in range(0, 1):

                real_volume_part = real_volume[kr, :, :, :, :]

                real_volume_part = torch.from_numpy(real_volume_part)
                real_volume_part = real_volume_part.squeeze()
                real_volume_part = real_volume_part.unsqueeze(1)
                real_volume_part = torch.concat([real_volume_part, real_volume_part, real_volume_part], dim=1)
                real_volume_int = (real_volume_part * 255.).type(torch.uint8).to(device)
                real_volume_part = (real_volume_part - 1.).to(device)

                fid.update(real_volume_int, real=True)
                fid.update(fake_volume_int, real=False)

                kid.update(real_volume_int, real=True)
                kid.update(fake_volume_int, real=False)

                KID_mean, _ = kid.compute()
                KID = KID + KID_mean
                FID = FID + fid.compute()
                LPIPS = LPIPS + lpips(real_volume_part, fake_volume_part).detach()
                print(str(i) + '_' + str(kf) + ' / ' + str(j) + ' : Finished' +
                      ', KID :' + str(KID) + ', FID : ' + str(FID) + ', LPIPS : ' + str(LPIPS))

IS = IS / (18. * float(res_batch_num))
KID = KID / (18. * 18. * float(res_batch_num))
FID = FID / (18. * float(res_batch_num))
LPIPS = LPIPS / (18. * 18. * float(res_batch_num))

print('IS : ' + str(IS))
print('KID : ' + str(KID.item()))
print('FID : ' + str(FID.item()))
print('LPIPS : ' + str(LPIPS.item()))