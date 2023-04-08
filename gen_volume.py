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

def gen_volume():

    class LOSS_DEC(Enum):
        NOT_USE = 1
        STEP = 2
        SMOOTH = 3

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=400, help="epoch to start training from")
    parser.add_argument("--dataset_name", type=str, default="KISTI_volume", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=128, help="size of image depth")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    opt = parser.parse_args()
    print(opt)

    volume_name = "volume_test2"
    model_name = "model_test2"

    os.makedirs("%s/%s" % (volume_name, opt.dataset_name), exist_ok=True)
    os.makedirs("%s/%s" % (model_name, opt.dataset_name), exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    use_ctsgan = False
    use_ctsgan_all = False

    # Initialize generator and discriminator
    encoder = EncodeInput()
    generator = GeneratorUNet()
    discriminator_volume = Discriminator_volume()
    if use_ctsgan or use_ctsgan_all:
        discriminator_slab = Discriminator_slab()
        discriminator_slices = Discriminator_slices()

    if cuda:
        encoder = encoder.cuda()
        generator = generator.cuda()
        discriminator_volume = discriminator_volume.cuda()
        if use_ctsgan or use_ctsgan_all:
            discriminator_slab = discriminator_slab.cuda()
            discriminator_slices = discriminator_slices.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        encoder.load_state_dict(torch.load("%s/encoder_%d.pth" % (model_name, opt.epoch)))
        generator.load_state_dict(torch.load("%s/generator_%d.pth" % (model_name, opt.epoch)))
        discriminator_volume.load_state_dict(torch.load("%s/discriminator_volume_%d.pth" % (model_name, opt.epoch)))
        if use_ctsgan or use_ctsgan_all:
            discriminator_slab.load_state_dict(torch.load("%s/discriminator_slab_%d.pth" % (model_name, opt.epoch)))
            discriminator_slices.load_state_dict(torch.load("%s/discriminator_slice_%d.pth" % (model_name, opt.epoch)))
    else:
        # Initialize weights
        sys.exit('ERROR: There is no weight assigned to that model folder and epoch number')


    torch.manual_seed(0)

    def sample_voxel_volumes(epoch, store):

        total_volume_num = 18
        for j in range(0, total_volume_num):

            # Model inputs
            encoder.eval()
            generator.eval()
            discriminator_volume.eval()
            if use_ctsgan or use_ctsgan_all:
                discriminator_slices.eval()
                discriminator_slab.eval()

            '''
            volume_noise = torch.rand(opt.batch_size, 128 * 128 * 128)
            volume_noise = volume_noise.reshape(opt.batch_size, 1, 128, 128, 128)
            volume_noise = volume_noise.cuda()
            enc_code = encoder(volume_noise)
            '''
            # enc_code = torch.clamp(enc_code, min=0.0, max=1.0)

            enc_noise = torch.randn(opt.batch_size * 262144)
            enc_noise = enc_noise.reshape(opt.batch_size, 512, 8, 8, 8)
            enc_noise = enc_noise.cuda()

            ref_code = enc_noise

            fake_volume = generator(ref_code)
            fake_volume = torch.clamp(fake_volume, min=0.0, max=1.0)

            # Print log
            sys.stdout.write(
                "\r[Test Epoch %d] [Batch %d/%d] Volume Sampled"
                % (
                    opt.epoch,
                    j,
                    total_volume_num,
                )
            )

            # convert to numpy arrays
            if store is True:

                store_folder = "%s/%s/epoch_%s_" % (volume_name, opt.dataset_name, epoch)
                np.save(store_folder + 'fake_EC_' + str(j).zfill(2) + '.npy', enc_noise.cpu().detach().numpy())

                fake_volume = fake_volume.cpu().detach().numpy()

                store_folder = "%s/%s/epoch_%s_" % (volume_name, opt.dataset_name, epoch)

                # np.save(store_folder + 'real_V_' + str(j).zfill(2) + '.npy', real_volume)
                np.save(store_folder + 'fake_V_' + str(j).zfill(2) + '.npy', fake_volume)


    sample_voxel_volumes(opt.epoch, True)
    print('*****volumes sampled*****')


if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    gen_volume()