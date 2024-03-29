import argparse
import os
import numpy as np
import time
import datetime
import sys
from enum import Enum

import evaluation_factor as ef

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from dataset import CTDataset

from dice_loss import diceloss

import torch
from torch import autograd

import binvox_rw


def calc_gradient_penalty(netD, real_data, fake_data, vol_or_not):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(1).cuda()
    if vol_or_not:
        alpha = alpha.expand(1, real_data.nelement()).contiguous().view(-1, 1, 128, 128, 128)
    else:
        alpha = alpha.expand(1, real_data.nelement()).contiguous().view(1, -1, 128, 128)

    # alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 1e-4

    return gradient_penalty


def train():

    class LOSS_DEC(Enum):
        NOT_USE = 1
        STEP = 2
        SMOOTH = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=410, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="KISTI_volume", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--elr", type=float, default=2e-6, help="adam: encoder learning rate") # Default : 2e-5
    parser.add_argument("--glr", type=float, default=2e-5, help="adam: generator learning rate") # Default : 2e-5
    parser.add_argument("--dlr", type=float, default=2e-5, help="adam: discriminator learning rate") # Default : 2e-5
    parser.add_argument("--elr_decay", type=float, default=2e-7, help="adam: encoder learning rate (decaying)")
    parser.add_argument("--glr_decay", type=float, default=2e-6, help="adam: generator learning rate (decaying)")
    parser.add_argument("--dlr_decay", type=float, default=2e-6, help="adam: discriminator learning rate (decaying)")
    parser.add_argument("--elr_decay2", type=float, default=2e-8, help="adam: encoder learning rate (decaying2)")
    parser.add_argument("--glr_decay2", type=float, default=2e-7, help="adam: generator learning rate (decaying2)")
    parser.add_argument("--dlr_decay2", type=float, default=2e-7, help="adam: discriminator learning rate (decaying2)")
    parser.add_argument("--elr_decay3", type=float, default=2e-7, help="adam: encoder learning rate (decaying3)")
    parser.add_argument("--glr_decay3", type=float, default=2e-7, help="adam: generator learning rate (decaying3)")
    parser.add_argument("--dlr_decay3", type=float, default=2e-7, help="adam: discriminator learning rate (decaying3)")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=9999, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=128, help="size of image depth")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--disc_update", type=int, default=5, help="only update discriminator every n iter")
    parser.add_argument("--d_threshold", type=int, default=.8, help="discriminator threshold")
    parser.add_argument("--d_iter", type=int, default=1, help="discriminator iteration number")
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument(
        "--sample_interval", type=int, default=80, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    volume_name = "volumes"
    model_name = "saved_models"

    os.makedirs("%s/%s" % (volume_name, opt.dataset_name), exist_ok=True)
    os.makedirs("%s/%s" % (model_name, opt.dataset_name), exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    L2_loss = torch.nn.MSELoss()
    criterion_voxelwise = diceloss()

    # Loss weight of L1 voxel-wise loss between translated image and real image
    # lambda_voxel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4, opt.img_depth // 2 ** 4)

    use_ctsgan = True
    use_slab = True
    use_slice = True

    # Initialize generator and discriminator
    encoder = EncodeInput()
    generator = GeneratorUNet()
    discriminator_volume = Discriminator_volume()
    if use_ctsgan:
        if use_slab:
            discriminator_slab = Discriminator_slab()
        if use_slice:
            discriminator_slices = Discriminator_slices()

    if cuda:
        encoder = encoder.cuda()
        generator = generator.cuda()
        discriminator_volume = discriminator_volume.cuda()
        if use_ctsgan:
            if use_slab:
                discriminator_slab = discriminator_slab.cuda()
            if use_slice:
                discriminator_slices = discriminator_slices.cuda()
        criterion_GAN.cuda()
        criterion_voxelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        encoder.load_state_dict(torch.load("%s/%s/encoder_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
        generator.load_state_dict(torch.load("%s/%s/generator_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
        discriminator_volume.load_state_dict(torch.load("%s/%s/discriminator_volume_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
        if use_ctsgan:
            if use_slab:
                discriminator_slab.load_state_dict(torch.load("%s/%s/discriminator_slab_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
            if use_slice:
                discriminator_slices.load_state_dict(torch.load("%s/%s/discriminator_slice_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        encoder.apply(weights_init_normal)
        generator.apply(weights_init_normal)
        discriminator_volume.apply(weights_init_normal)
        if use_ctsgan:
            if use_slab:
                discriminator_slab.apply(weights_init_normal)
            if use_slice:
                discriminator_slices.apply(weights_init_normal)

    # Optimizers
    optimizer_E = torch.optim.RMSprop(encoder.parameters(), lr=opt.elr)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.glr)
    optimizer_DV = torch.optim.RMSprop(discriminator_volume.parameters(), lr=opt.dlr)

    dataloader = DataLoader(
        CTDataset("./data/liver/", None),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(0)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    discriminator_update = 'False'
    slab_size = 28
    slice_num = 56
    for epoch in range(opt.epoch, opt.n_epochs):

        encoder.train()
        generator.train()
        discriminator_volume.train()

        for i, batch in enumerate(dataloader):

            # Random weight generation for code & noise

            w_enc_code = torch.rand(1)
            w_enc_noise = 1 - w_enc_code            

            '''
            w_enc_code = 0.5 * torch.rand(1)
            w_enc_noise = 1 - w_enc_code
            '''
            
            w_enc_code = w_enc_code.cuda()
            w_enc_noise = w_enc_noise.cuda()

            '''
            w_enc_code = 0.5
            w_enc_noise = 0.5
            '''

            real_volume = Variable(batch["B"].unsqueeze_(1).type(Tensor))
            real_volume = real_volume / 255.
            real_volume_max = real_volume.max()
            real_volume_min = real_volume.min()
            real_volume = (real_volume - real_volume_min) / (real_volume_max - real_volume_min)
            real_volume = torch.clamp(real_volume, min=0.0, max=1.0)

            # --------------------
            #  Train Discriminator
            # --------------------
            for di in range(0, opt.d_iter):
                optimizer_G.zero_grad()
                optimizer_E.zero_grad()
                optimizer_DV.zero_grad()

                # ---------------------
                #  Train Discriminator, only update every disc_update batches
                # ---------------------
                # Real loss

                real_volume_input = real_volume.reshape(opt.batch_size, 1, 128, 128, 128)
                '''
                volume_noise = torch.rand(opt.batch_size * 128 * 128 * 128)
                volume_noise = volume_noise.reshape(opt.batch_size, 1, 128, 128, 128)
                volume_noise = volume_noise.cuda()
                '''
                enc_code = encoder(real_volume_input)

                enc_noise = torch.randn(opt.batch_size * 262144)
                enc_noise = enc_noise.reshape(opt.batch_size, 512, 8, 8, 8)
                enc_noise = enc_noise.cuda()

                input_code = w_enc_code * enc_code + w_enc_noise * enc_noise
                fake_volume = generator(input_code)

                fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)

                pred_real_volume = discriminator_volume(real_volume)
                pred_fake_volume = discriminator_volume(fake_volume.detach())

                # DV_loss = 0.5 * (DV_loss_real_volume + DV_loss_fake_volume)
                DV_loss = - torch.mean(pred_real_volume) + torch.mean(pred_fake_volume)

                DSLB_loss = 0.
                DSLC_loss = 0.
                if use_ctsgan:

                    # Adversarial ground truths
                    # valid = torch.ones([1, 1, 8, 8], requires_grad=False).cuda()
                    # fake = torch.zeros([1, 1, 8, 8], requires_grad=False).cuda()

                    # Extract slab from volume
                    if use_slab:
                        slab_position = torch.randint(int(slab_size / 2), 127 - int(slab_size / 2), (4, ))

                        slab_range = torch.zeros(4, 2)
                        for j in range(0, 4):
                            slab_range[j, 0] = slab_position[j] - int(slab_size / 2)
                            slab_range[j, 1] = slab_position[j] + int(slab_size / 2)

                        real_volume_slab = []
                        fake_volume_slab = []
                        for j in range(0, 4):
                            if opt.batch_size > 1:
                                real_volume_slab.append(real_volume[0, 0:opt.batch_size-1, :, :, int(slab_range[j, 0]):int(slab_range[j, 1])])
                                fake_volume_slab.append(fake_volume[0, 0:opt.batch_size-1, :, :, int(slab_range[j, 0]):int(slab_range[j, 1])])
                            else:
                                real_volume_slab.append(real_volume[0, 0, :, :, int(slab_range[j, 0]):int(slab_range[j, 1])])
                                fake_volume_slab.append(fake_volume[0, 0, :, :, int(slab_range[j, 0]):int(slab_range[j, 1])])

                        real_volume_slab_tot = torch.cat([real_volume_slab[0], real_volume_slab[1], real_volume_slab[2],
                                                          real_volume_slab[3]], dim=2).reshape(1, -1, 128, 128)
                        fake_volume_slab_tot = torch.cat([fake_volume_slab[0], fake_volume_slab[1], fake_volume_slab[2],
                                                          fake_volume_slab[3]], dim=2).reshape(1, -1, 128, 128)

                        pred_real_slab = discriminator_slab(real_volume_slab_tot.detach())
                        pred_fake_slab = discriminator_slab(fake_volume_slab_tot.detach())

                        DSLB_loss = - torch.mean(pred_real_slab) + torch.mean(pred_fake_slab)

                    # Extract slices from volume
                    if use_slice:
                        slices_position = torch.randint(1, 126, (slice_num,))

                        slices_range = torch.zeros(slice_num, 2)
                        for j in range(0, slice_num):
                            slices_range[j, 0] = slices_position[j] - 1
                            slices_range[j, 1] = slices_position[j] + 1

                        real_volume_slices = []
                        fake_volume_slices = []
                        for j in range(0, slice_num):
                            if opt.batch_size > 1:
                                real_volume_slices.append(real_volume[0, 0:opt.batch_size-1, :, :, int(slices_range[j, 0]):int(slices_range[j, 1])])
                                fake_volume_slices.append(fake_volume[0, 0:opt.batch_size-1, :, :, int(slices_range[j, 0]):int(slices_range[j, 1])])
                            else:
                                real_volume_slices.append(real_volume[0, 0, :, :, int(slices_range[j, 0]):int(slices_range[j, 1])])
                                fake_volume_slices.append(fake_volume[0, 0, :, :, int(slices_range[j, 0]):int(slices_range[j, 1])])

                        real_volume_slices_tot = torch.cat(real_volume_slices, dim=2).reshape(1, -1, 128, 128)
                        fake_volume_slices_tot = torch.cat(fake_volume_slices, dim=2).reshape(1, -1, 128, 128)

                        pred_real_slices = discriminator_slices(real_volume_slices_tot.detach())
                        pred_fake_slices = discriminator_slices(fake_volume_slices_tot.detach())

                        DSLC_loss = - torch.mean(pred_real_slices) + torch.mean(pred_fake_slices)

                    Disc_loss = DV_loss + DSLB_loss + DSLC_loss
                    Disc_loss = Disc_loss / 3.
                else:
                    Disc_loss = DV_loss

                # D_loss = 5. * D_loss

                loss_dist = L1_loss(fake_volume, real_volume)
                loss_uqi = 1. - ef.uqi_volume(fake_volume.cpu().detach().numpy(), real_volume.cpu().detach().numpy(), normalize=True)
                cont_loss = 0.0
                for j in range(1, 126):
                    if opt.batch_size > 1:
                        fake_volume_slice = fake_volume[0, 0:opt.batch_size-1, :, :, j]
                        fake_volume_slice_after = fake_volume[0, 0:opt.batch_size-1, :, :, j + 1]
                        real_volume_slice = real_volume[0, 0:opt.batch_size-1, :, :, j]
                        real_volume_slice_after = real_volume[0, 0:opt.batch_size-1, :, :, j + 1]
                    else:
                        fake_volume_slice = fake_volume[0, 0, :, :, j]
                        fake_volume_slice_after = fake_volume[0, 0, :, :, j + 1]
                        real_volume_slice = real_volume[0, 0, :, :, j]
                        real_volume_slice_after = real_volume[0, 0, :, :, j + 1]
                    fvs_diff = L1_loss(fake_volume_slice, fake_volume_slice_after)
                    rvs_diff = L1_loss(real_volume_slice, real_volume_slice_after)
                    cont_loss = cont_loss + torch.abs(fvs_diff - rvs_diff)
                cont_loss = cont_loss / 126.

                weight_recon = 1.
                D_loss = Disc_loss + weight_recon * loss_dist.item() + weight_recon * loss_uqi.item() + weight_recon * cont_loss.item()

                gp_vol = calc_gradient_penalty(discriminator_volume, real_volume.data, fake_volume.data, vol_or_not=True)
                # gp_slab = calc_gradient_penalty(discriminator_slab, real_volume_slab_tot.data, fake_volume_slab_tot.data, vol_or_not=False)
                # gp_slice = calc_gradient_penalty(discriminator_slices, real_volume_slices_tot.data, fake_volume_slices_tot.data, vol_or_not=False)
                # gp_value = (gp_vol + gp_slab + gp_slice) / 3.
                gp_value = gp_vol

                # print(pred_real_volume.shape)
                # print(pred_fake_volume.shape)

                d_real_acu_volume = torch.ge(pred_real_volume.squeeze(), 0.5).float()
                d_fake_acu_volume = torch.le(pred_fake_volume.squeeze(), 0.5).float()
                d_total_acu_volume = torch.mean(torch.cat((d_real_acu_volume, d_fake_acu_volume), 0))

                d_total_acu_slab = 0.
                d_total_acu_slices = 0.
                if use_ctsgan:

                    if use_slab:
                        d_real_acu_slab = torch.ge(pred_real_slab.squeeze(), 0.5).float()
                        d_fake_acu_slab = torch.le(pred_fake_slab.squeeze(), 0.5).float()
                        d_total_acu_slab = torch.mean(torch.cat((d_real_acu_slab, d_fake_acu_slab), 0))

                    if use_slice:
                        d_real_acu_slices = torch.ge(pred_real_slices.squeeze(), 0.5).float()
                        d_fake_acu_slices = torch.le(pred_fake_slices.squeeze(), 0.5).float()
                        d_total_acu_slices = torch.mean(torch.cat((d_real_acu_slices, d_fake_acu_slices), 0))

                    # d_total_acu_slices = (pred_real_slices + pred_fake_slices) / 2.

                    d_total_acu = (d_total_acu_volume + d_total_acu_slab + d_total_acu_slices) / 3.
                else:
                    d_total_acu = d_total_acu_volume

                if d_total_acu <= opt.d_threshold:
                    optimizer_DV.zero_grad()
                    '''
                    if use_ctsgan or use_ctsgan_all:
                        optimizer_DSL.zero_grad()
                        optimizer_DSC.zero_grad()
                        '''
                    D_loss.backward()
                    gp_value.backward() # Gradient Penalty (WGAN-GP)
                    optimizer_DV.step()
                    # scheduler_DV.step()
                    '''
                    if use_ctsgan or use_ctsgan_all:
                        optimizer_DSL.step()
                        optimizer_DSC.step()
                        '''
                    discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_DV.zero_grad()
            optimizer_E.zero_grad()
            '''
            if use_ctsgan or use_ctsgan_all:
                optimizer_DSL.zero_grad()
                optimizer_DSC.zero_grad()
                '''
            optimizer_G.zero_grad()

            # Adversarial ground truths
            valid = torch.ones([1, 1, 8, 8, 8], requires_grad=False).cuda()

            real_volume_input = real_volume.reshape(opt.batch_size, 1, 128, 128, 128)
            '''
            volume_noise = torch.rand(opt.batch_size * 128 * 128 * 128)
            volume_noise = volume_noise.reshape(opt.batch_size, 1, 128, 128, 128)
            volume_noise = volume_noise.cuda()
            '''

            enc_code = encoder(real_volume_input)

            enc_noise = torch.randn(opt.batch_size * 262144)
            enc_noise = enc_noise.reshape(opt.batch_size, 512, 8, 8, 8)
            enc_noise = enc_noise.cuda()

            input_code = w_enc_code * enc_code + w_enc_noise * enc_noise
            fake_volume = generator(input_code)
            fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)

            GAN_loss = -torch.mean(discriminator_volume(fake_volume))

            cont_loss = 0.0
            for j in range (1, 126):
                fake_volume_slice = fake_volume[0, 0, :, :, j]
                fake_volume_slice_after = fake_volume[0, 0, :, :, j+1]
                # real_volume_slice = real_volume[0, 0, :, :, j]
                # real_volume_slice_after = real_volume[0, 0, :, :, j+1]
                real_volume_slice = real_volume[0, 0, :, :, j]
                real_volume_slice_after = real_volume[0, 0, :, :, j + 1]
                fvs_diff = L1_loss(fake_volume_slice, fake_volume_slice_after)
                rvs_diff = L1_loss(real_volume_slice, real_volume_slice_after)
                cont_loss = cont_loss + torch.abs(fvs_diff - rvs_diff)
            cont_loss = cont_loss / 126.

            # Regularization Factors (L1, UQI, IoU)
            # loss_dist = L1_loss(fake_volume, real_volume)
            loss_dist = L1_loss(fake_volume, real_volume)
            # loss_uqi = 1. - ef.uqi_volume(fake_volume.cpu().detach().numpy(), real_volume.cpu().detach().numpy(), normalize=True)
            loss_uqi = 1. - ef.uqi_volume(fake_volume.cpu().detach().numpy(), real_volume.cpu().detach().numpy(), normalize=True)

            # IoU per sample
            sample_iou = []
            for th in [.46]:
                # for th in [0.3]:
                _volume = torch.ge(fake_volume, th).float()
                # _gt_volume = torch.ge(real_volume, th).float()
                _gt_volume = torch.ge(real_volume, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            iou_loss = sum(sample_iou) / len(sample_iou)
            iou_loss = 1. - iou_loss

            weight_recon_start = 1.
            weight_recon = weight_recon_start
            weight_recon_min = 0.
            epoch_start = 250
            loss_dec_state = LOSS_DEC.NOT_USE
            # iou_loss = iou_loss - 0.5
            if loss_dec_state is LOSS_DEC.NOT_USE:
                G_loss = GAN_loss + weight_recon * loss_dist.item() + weight_recon * cont_loss.item() + weight_recon * loss_uqi.item() # + weight_recon * iou_loss
            elif loss_dec_state is LOSS_DEC.STEP:
                step_epoch = 10.
                total_level = round(opt.n_epochs / step_epoch)
                cur_level = round(epoch / step_epoch)
                dec_factor = weight_recon / total_level * cur_level
                weight_recon = weight_recon - dec_factor
                if weight_recon > weight_recon_min:
                    G_loss = GAN_loss + weight_recon * loss_dist.item() + weight_recon * cont_loss.item() + weight_recon * loss_uqi.item() # + weight_recon * iou_loss
                else:
                    G_loss = GAN_loss
            elif loss_dec_state is LOSS_DEC.SMOOTH:
                if epoch >= epoch_start:
                    dec_factor = weight_recon_start / (opt.n_epochs - epoch_start) * (epoch - epoch_start)
                    # dec_factor = dec_factor * 0.5
                    weight_recon = weight_recon_start - dec_factor

                if weight_recon > weight_recon_min:
                    G_loss = GAN_loss + weight_recon * loss_dist.item() + weight_recon * cont_loss.item() + weight_recon * loss_uqi.item() # + weight_recon * iou_loss
                else:
                    G_loss = GAN_loss
            else:
                print('Error Occurred : Decrement State is not defined')
                exit(-1)

            G_loss.backward()
            optimizer_G.step()
            # scheduler_G.step()
            # optimizer_E.step()

            # ------------------
            #  Train Encoders
            # ------------------

            optimizer_DV.zero_grad()
            optimizer_G.zero_grad()
            '''
            if use_ctsgan or use_ctsgan_all:
                optimizer_DSL.zero_grad()
                optimizer_DSC.zero_grad()
                '''
            optimizer_E.zero_grad()
            '''
            volume_noise = torch.rand(128 * 128 * 128)
            volume_noise = volume_noise.reshape(1, 1, 128, 128, 128)
            volume_noise = volume_noise.cuda()
            input_code = encoder(volume_noise)
            fake_volume = generator(input_code)
            fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)
            '''

            real_volume_input = real_volume.reshape(opt.batch_size, 1, 128, 128, 128)
            '''
            volume_noise = torch.rand(opt.batch_size * 128 * 128 * 128)
            volume_noise = volume_noise.reshape(opt.batch_size, 1, 128, 128, 128)
            volume_noise = volume_noise.cuda()
            '''

            # real_volume_input = 0.2 * real_volume_input + 0.8 * volume_noise
            # ref_code = encoder(real_volume_input)

            ref_code = encoder(real_volume_input)
            '''
            enc_noise = torch.randn(opt.batch_size * 262144)
            enc_noise = enc_noise.reshape(opt.batch_size, 512, 8, 8, 8)
            enc_noise = enc_noise.cuda()

            ref_code = 0.5 * ref_code + 0.5 * enc_noise
            '''
            real_volume_decode = generator(ref_code)
            real_volume_decode = torch.clamp(real_volume_decode, min=0.0, max=1.0)

            # print(real_volume.shape)
            # print(real_volume_decode.shape)

            weight_recon_enc = 1.
            enc_loss = L1_loss(real_volume, real_volume_decode)
            E_loss = enc_loss # + weight_recon_enc * loss_dist.item() + weight_recon_enc * loss_uqi.item() + weight_recon_enc * cont_loss.item()

            E_loss.backward()
            optimizer_E.step()
            # scheduler_E.step()

            batches_done = epoch * len(dataloader) + i

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log & [E loss : %f]
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [E loss: %f lr: %e] [G loss: %f, lr: %e, adv: %f, L1: %f, iou: %f, sim: %f, cont: %f,w: %f] [D loss: %f, lr: %e, GP value: %f, D accuracy: %f, D update: %s] ETA: %s"
                # "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, adv: %f, L1: %f, iou: %f, sim: %f, cont: %f,w: %f] [D loss: %f, D accuracy: %f, D update: %s] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    E_loss.item(),
                    optimizer_E.param_groups[0]['lr'],
                    G_loss.item(),
                    optimizer_G.param_groups[0]['lr'],
                    GAN_loss.item(),
                    loss_dist.item(),
                    iou_loss,
                    loss_uqi.item(),
                    cont_loss.item(),
                    weight_recon,
                    D_loss.item(),
                    optimizer_DV.param_groups[0]['lr'],
                    gp_value.item(),
                    # 0.0,
                    d_total_acu,
                    discriminator_update,
                    time_left,
                )
            )
            discriminator_update = 'False'

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch > 0:
            # Save model checkpoints
            torch.save(encoder.state_dict(), "%s/%s/encoder_%d.pth" % (model_name, opt.dataset_name, epoch))
            torch.save(generator.state_dict(), "%s/%s/generator_%d.pth" % (model_name, opt.dataset_name, epoch))
            torch.save(discriminator_volume.state_dict(), "%s/%s/discriminator_volume_%d.pth" % (model_name, opt.dataset_name, epoch))
            if use_ctsgan:
                if use_slab:
                    torch.save(discriminator_slab.state_dict(), "%s/%s/discriminator_slab_%d.pth" % (model_name, opt.dataset_name, epoch))
                if use_slice:
                    torch.save(discriminator_slices.state_dict(), "%s/%s/discriminator_slice_%d.pth" % (model_name, opt.dataset_name, epoch))

        print(' *****training processed*****')

        # If at sample interval save image
        '''
        if epoch % opt.sample_interval == 0 and epoch > 0:
            sample_voxel_volumes(epoch, True)
            print('*****volumes sampled*****')
        else:
            sample_voxel_volumes(epoch, False)
            print(' *****testing processed*****')
            '''


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    train()
1