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

from model.model_3dgan import *
from dataset import CTDataset

from dice_loss import diceloss

import torch

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="KISTI_volume", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--glr", type=float, default=2e-6, help="adam: generator learning rate") # Default : 2e-5
    parser.add_argument("--dlr", type=float, default=2e-6, help="adam: discriminator learning rate") # Default : 2e-5
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
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument(
        "--sample_interval", type=int, default=50, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    volume_name = "volumes2"
    model_name = "saved_models2"

    os.makedirs("%s/%s" % (volume_name, opt.dataset_name), exist_ok=True)
    os.makedirs("%s/%s" % (model_name, opt.dataset_name), exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()
    criterion_voxelwise = diceloss()

    use_ctsgan = True
    use_ctsgan_all = False

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_voxelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("%s/%s/generator_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("%s/%s/discriminator_%d.pth" % (model_name, opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.glr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.dlr, betas=(opt.b1, opt.b2))

    dataloader = DataLoader(
        CTDataset("./data/orig/train/", None),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    '''
    val_dataloader = DataLoader(
        CTDataset("./data/orig/test/", None),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )
    '''

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(0)

    def sample_voxel_volumes(epoch, store):

        total_volume_num = 6
        for j in range(0, total_volume_num):

            # Model inputs
            generator.eval()
            discriminator.eval()

            volume_noise = torch.rand(opt.batch_size, 200, 1, 1, 1).cuda()
            fake_volume = generator(volume_noise)
            fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)

            # Print log
            sys.stdout.write(
                "\r[Test Epoch %d/%d] [Batch %d/%d] Volume Sampled"
                % (
                    epoch,
                    opt.n_epochs,
                    j,
                    total_volume_num,
                )
            )

            # convert to numpy arrays
            if store is True:
                fake_volume = fake_volume.cpu().detach().numpy()

                store_folder = "%s/%s/epoch_%s_" % (volume_name, opt.dataset_name, epoch)

                # np.save(store_folder + 'real_V_' + str(j).zfill(2) + '.npy', real_volume)
                np.save(store_folder + 'fake_V_' + str(j).zfill(2) + '.npy', fake_volume)

    # ----------
    #  Training
    # ----------

    prev_time = time.time()
    discriminator_update = 'False'
    for epoch in range(opt.epoch, opt.n_epochs):

        generator.train()
        discriminator.train()

        for i, batch in enumerate(dataloader):

            # --------------------
            #  Train Discriminator
            # --------------------
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            real_volume = Variable(batch["B"].unsqueeze_(1).type(Tensor))
            real_volume = real_volume / 255.
            real_volume_max = real_volume.max()
            real_volume_min = real_volume.min()
            real_volume = (real_volume - real_volume_min) / (real_volume_max - real_volume_min)
            real_volume = torch.clamp(real_volume, min=0.0, max=1.0)

            # ---------------------
            #  Train Discriminator, only update every disc_update batches
            # ---------------------
            # Real loss
            volume_noise = torch.rand(opt.batch_size, 200, 1, 1, 1).cuda()
            fake_volume = generator(volume_noise)
            fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)

            pred_real_volume = discriminator(real_volume)
            pred_fake_volume = discriminator(fake_volume.detach())

            '''
            # Adversarial ground truths
            valid = torch.ones_like(pred_real_volume).cuda()
            fake = torch.zeros_like(pred_fake_volume).cuda()

            # Calculate Volume Discriminator Loss
            DV_loss_real_volume = criterion_GAN(pred_real_volume, valid)
            DV_loss_fake_volume = criterion_GAN(pred_fake_volume, fake)

            D_loss = 0.5 * (DV_loss_real_volume + DV_loss_fake_volume)
            '''

            # For WGAN
            D_loss = - torch.mean(pred_real_volume) + torch.mean(pred_fake_volume)

            '''
            d_real_acu_volume = torch.ge(pred_real_volume.squeeze(), 0.5).float()
            d_fake_acu_volume = torch.le(pred_fake_volume.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu_volume, d_fake_acu_volume), 0))

            if d_total_acu <= opt.d_threshold:
            '''
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()
            discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()


            volume_noise = torch.rand(opt.batch_size, 200, 1, 1, 1).cuda()
            fake_volume = generator(volume_noise)
            fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)

            pred_fake_volume = discriminator(fake_volume)

            '''
            # Adversarial ground truths
            valid = torch.ones_like(pred_fake_volume).cuda()

            G_loss = criterion_GAN(pred_fake_volume, valid)
            '''

            # For WGAN
            G_loss = -torch.mean(discriminator(fake_volume))

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

            # Print log & [E loss : %f]
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f] ETA: %s"
                # "\r[Epoch %d/%d] [Batch %d/%d] [G loss: %f, adv: %f, L1: %f, iou: %f, sim: %f, cont: %f,w: %f] [D loss: %f, D accuracy: %f, D update: %s] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    G_loss.item(),
                    D_loss.item(),
                    time_left,
                )
            )
            discriminator_update = 'False'

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch > 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "%s/%s/generator_%d.pth" % (model_name, opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "%s/%s/discriminator_%d.pth" % (model_name, opt.dataset_name, epoch))

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    train()
