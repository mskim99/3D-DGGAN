import argparse
import os
import numpy as np
import time
import datetime
import sys

import evaluation_factor as ef

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from dataset import CTDataset

from dice_loss import diceloss

import torch

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=210, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="KISTI_volume", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--glr", type=float, default=2e-5, help="adam: generator learning rate") # Default : 2e-4
    parser.add_argument("--dlr", type=float, default=2e-5, help="adam: discriminator learning rate") # Default : 2e-4
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=128, help="size of image depth")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--disc_update", type=int, default=5, help="only update discriminator every n iter")
    parser.add_argument("--d_threshold", type=int, default=.8, help="discriminator threshold")
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument(
        "--sample_interval", type=int, default=200, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=200, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("volumes4/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models4/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    criterion_voxelwise = diceloss()

    # Loss weight of L1 voxel-wise loss between translated image and real image
    # lambda_voxel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4, opt.img_depth // 2 ** 4)

    use_ctsgan = False

    # Initialize generator and discriminator
    generator = GeneratorUNet()
    discriminator_volume = Discriminator_volume()
    if use_ctsgan:
        discriminator_slab = Discriminator_slab()
        discriminator_slices = Discriminator_slices()

    if cuda:
        generator = generator.cuda()
        discriminator_volume = discriminator_volume.cuda()
        if use_ctsgan:
            discriminator_slab = discriminator_slab.cuda()
            discriminator_slices = discriminator_slices.cuda()
        criterion_GAN.cuda()
        criterion_voxelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models4/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator_volume.load_state_dict(torch.load("saved_models4/%s/discriminator_volume_%d.pth" % (opt.dataset_name, opt.epoch)))
        if use_ctsgan:
            discriminator_slab.load_state_dict(torch.load("saved_models4/%s/discriminator_slab_%d.pth" % (opt.dataset_name, opt.epoch)))
            discriminator_slices.load_state_dict(torch.load("saved_models4/%s/discriminator_slice_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator_volume.apply(weights_init_normal)
        if use_ctsgan:
            discriminator_slab.apply(weights_init_normal)
            discriminator_slices.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.b1, opt.b2))
    optimizer_DV = torch.optim.Adam(discriminator_volume.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))
    if use_ctsgan:
        optimizer_DSL = torch.optim.Adam(discriminator_slab.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))
        optimizer_DSC = torch.optim.Adam(discriminator_slices.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))

    dataloader = DataLoader(
        CTDataset("./data/orig/train/", None),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    val_dataloader = DataLoader(
        CTDataset("./data/orig/test/", None),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(0)

    def sample_voxel_volumes(epoch, store):

        for j, batch_test in enumerate(val_dataloader):

            # Model inputs
            volume_noise = torch.randn(262144)
            volume_noise = volume_noise.reshape(1, 512, 8, 8, 8)
            volume_noise = volume_noise.cuda()
            fake_volume = generator(volume_noise)

            '''
            real_volume = Variable(batch_test["B"].unsqueeze_(1).type(Tensor))
            real_volume = real_volume / 255.
            real_volume_max = real_volume.max()
            real_volume_min = real_volume.min()
            real_volume = (real_volume - real_volume_min) / (real_volume_max - real_volume_min)
            '''

            # Print log
            sys.stdout.write(
                "\r[Test Epoch %d/%d] [Batch %d/%d] Volume Sampled"
                % (
                    epoch,
                    opt.n_epochs,
                    j,
                    len(val_dataloader),
                )
            )

            # convert to numpy arrays
            if store is True:
                # real_A = real_A.cpu().detach().numpy()
                # real_volume = real_volume.cpu().detach().numpy()
                fake_volume = fake_volume.cpu().detach().numpy()

                store_folder = "volumes4/%s/epoch_%s_" % (opt.dataset_name, epoch)

                # np.save(store_folder + 'real_V_' + str(j).zfill(2) + '.npy', real_volume)
                np.save(store_folder + 'fake_V_' + str(j).zfill(2) + '.npy', fake_volume)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    discriminator_update = 'False'
    slab_size = 28
    for epoch in range(opt.epoch, opt.n_epochs):

        for i, batch in enumerate(dataloader):

            # Model inputs
            volume_noise = torch.randn(262144)
            volume_noise = volume_noise.reshape(1, 512, 8, 8, 8)
            volume_noise = volume_noise.cuda()

            # Adversarial ground truths
            valid = torch.ones((volume_noise.size(0), 1), requires_grad=False).cuda()
            fake = torch.zeros((volume_noise.size(0), 1), requires_grad=False).cuda()

            real_volume = Variable(batch["B"].unsqueeze_(1).type(Tensor))
            real_volume = real_volume / 255.
            real_volume_max = real_volume.max()
            real_volume_min = real_volume.min()
            real_volume = (real_volume - real_volume_min) / (real_volume_max - real_volume_min)

            # ---------------------
            #  Train Discriminator, only update every disc_update batches
            # ---------------------
            # Real loss
            fake_volume = generator(volume_noise)
            pred_real_volume = discriminator_volume(real_volume)
            pred_fake_volume = discriminator_volume(fake_volume.detach())

            # Calculate Volume Discriminator Loss
            DV_loss_real_volume = criterion_GAN(pred_real_volume, valid)
            DV_loss_fake_volume = criterion_GAN(pred_fake_volume, fake)

            DV_loss = 0.5 * (DV_loss_real_volume + DV_loss_fake_volume)


            if use_ctsgan:
                # Extract slab from volume
                slab_position = torch.randint(int(slab_size / 2), 127 - int(slab_size / 2), (4, ))

                slab_range = torch.zeros(4, 2)
                for i in range(0, 4):
                    slab_range[i, 0] = slab_position[i] - int(slab_size / 2)
                    slab_range[i, 1] = slab_position[i] + int(slab_size / 2)

                real_volume_slab = []
                fake_volume_slab = []
                for i in range(0, 4):
                    real_volume_slab.append(real_volume[0, 0, :, :, int(slab_range[i, 0]):int(slab_range[i, 1])])
                    fake_volume_slab.append(fake_volume[0, 0, :, :, int(slab_range[i, 0]):int(slab_range[i, 1])])

                real_volume_slab_tot = torch.cat([real_volume_slab[0], real_volume_slab[1], real_volume_slab[2],
                                                  real_volume_slab[3]], dim=2).reshape(1, -1, 128, 128)
                fake_volume_slab_tot = torch.cat([fake_volume_slab[0], fake_volume_slab[1], fake_volume_slab[2],
                                                  fake_volume_slab[3]], dim=2).reshape(1, -1, 128, 128)

                pred_real_slab = discriminator_slab(real_volume_slab_tot.detach())
                pred_fake_slab = discriminator_slab(fake_volume_slab_tot.detach())

                # Calculate Slab Discriminator Loss
                DV_loss_real_slab = criterion_GAN(pred_real_slab, valid)
                DV_loss_fake_slab = criterion_GAN(pred_fake_slab, fake)

                DSLB_loss = 0.5 * (DV_loss_real_slab + DV_loss_fake_slab)


                # Extract slices from volume
                slices_position = torch.randint(1, 126, (28,))

                slices_range = torch.zeros(28, 2)
                for i in range(0, 28):
                    slices_range[i, 0] = slices_position[i] - 1
                    slices_range[i, 1] = slices_position[i] + 2

                real_volume_slices = []
                fake_volume_slices = []
                for i in range(0, 28):
                    real_volume_slices.append(real_volume[0, 0, :, :, int(slices_range[i, 0]):int(slices_range[i, 1])])
                    fake_volume_slices.append(fake_volume[0, 0, :, :, int(slices_range[i, 0]):int(slices_range[i, 1])])

                real_volume_slices_tot = torch.cat(real_volume_slices, dim=2).reshape(1, -1, 128, 128)
                fake_volume_slices_tot = torch.cat(real_volume_slices, dim=2).reshape(1, -1, 128, 128)

                pred_real_slices = discriminator_slices(real_volume_slices_tot.detach())
                pred_fake_slices = discriminator_slices(fake_volume_slices_tot.detach())

                # Calculate Slice Discriminator Loss
                DV_loss_real_slices = criterion_GAN(pred_real_slices, valid)
                DV_loss_fake_slices = criterion_GAN(pred_fake_slices, fake)

                DSLC_loss = 0.5 * (DV_loss_real_slices + DV_loss_fake_slices)

                D_loss = DV_loss + DSLB_loss + DSLC_loss
            else:
                D_loss = DV_loss

            d_real_acu_volume = torch.ge(pred_real_volume.squeeze(), 0.5).float()
            d_fake_acu_volume = torch.le(pred_fake_volume.squeeze(), 0.5).float()
            d_total_acu_volume = torch.mean(torch.cat((d_real_acu_volume, d_fake_acu_volume), 0))

            if use_ctsgan:
                d_real_acu_slab = torch.ge(pred_real_slab.squeeze(), 0.5).float()
                d_fake_acu_slab = torch.le(pred_fake_slab.squeeze(), 0.5).float()
                d_total_acu_slab = torch.mean(torch.cat((d_real_acu_slab, d_fake_acu_slab), 0))

                d_real_acu_slices = torch.ge(pred_real_slices.squeeze(), 0.5).float()
                d_fake_acu_slices = torch.le(pred_fake_slices.squeeze(), 0.5).float()
                d_total_acu_slices = torch.mean(torch.cat((d_real_acu_slices, d_fake_acu_slices), 0))

                d_total_acu = (d_total_acu_volume + d_total_acu_slab + d_total_acu_slices) / 3.
            else:
                d_total_acu = d_total_acu_volume

            if d_total_acu <= opt.d_threshold:
                optimizer_DV.zero_grad()
                if use_ctsgan:
                    optimizer_DSL.zero_grad()
                    optimizer_DSC.zero_grad()
                D_loss.backward()
                optimizer_DV.step()
                if use_ctsgan:
                    optimizer_DSL.step()
                    optimizer_DSC.step()
                discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_DV.zero_grad()
            if use_ctsgan:
                optimizer_DSL.zero_grad()
                optimizer_DSC.zero_grad()
            optimizer_G.zero_grad()

            fake_volume = generator(volume_noise)
            pred_fake_volume = discriminator_volume(fake_volume)
            GAN_loss = criterion_GAN(pred_fake_volume, valid)

            # Regularization Factors (L1, UQI, IoU)
            loss_L1 = L1_loss(fake_volume, real_volume)
            loss_uqi = 1. - ef.uqi_volume(fake_volume.cpu().detach().numpy(), real_volume.cpu().detach().numpy(), normalize=True)

            # IoU per sample
            sample_iou = []
            for th in [.2, .3, .4, .5]:
                # for th in [0.3]:
                _volume = torch.ge(fake_volume, th).float()
                _gt_volume = torch.ge(real_volume, th).float()
                intersection = torch.sum(torch.ge(_volume.mul(_gt_volume), 1)).float()
                union = torch.sum(torch.ge(_volume.add(_gt_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            iou_loss = sum(sample_iou) / len(sample_iou)
            iou_loss = 1. - iou_loss

            G_loss = GAN_loss + 10. * loss_L1 # + 3.3 * iou_loss + 3.3 * loss_uqi

            G_loss.backward()
            optimizer_G.step()

            batches_done = epoch * len(dataloader) + i

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f, D accuracy: %f, D update: %s] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    G_loss.item(),
                    D_loss.item(),
                    d_total_acu,
                    discriminator_update,
                    time_left,
                )
            )

            discriminator_update = 'False'

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models4/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator_volume.state_dict(), "saved_models4/%s/discriminator_volume_%d.pth" % (opt.dataset_name, epoch))
            if use_ctsgan:
                torch.save(discriminator_slab.state_dict(), "saved_models4/%s/discriminator_slab_%d.pth" % (opt.dataset_name, epoch))
                torch.save(discriminator_slices.state_dict(), "saved_models4/%s/discriminator_slice_%d.pth" % (opt.dataset_name, epoch))

        print(' *****training processed*****')

        # If at sample interval save image
        if epoch % opt.sample_interval == 0 and epoch > 0:
            sample_voxel_volumes(epoch, True)
            print('*****volumes sampled*****')
        else:
            sample_voxel_volumes(epoch, False)
            print(' *****testing processed*****')


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    train()
