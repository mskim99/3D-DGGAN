import vtk
import numpy as np
from vtk.util import numpy_support
import os
import binvox_rw
from scipy.ndimage import zoom

name = "230505_4_log_E_no_norm_lr_2e_6_G_ns_ref_rand_w_1_D_sgan_slice_76_recon_no_IoU_cont_epoch_410_WGAN_GP_3_RMSprop_batch_size_4_orig_seg_check"

for idx in range(400, 440, 40):
    for i in range (0, 1):
        data = np.load('J:/Program/CT_VSGAN_data/gen_volume/' + name + '/epoch_' + str(idx) + '_fake_V_' + str(i).zfill(2) + '.npy')
        # print(data.shape)

        for j in range (0, 4):
            data_portion = data[j, 0, :, :, :]
            # data = data / 255.
            imdata = vtk.vtkImageData()

            depthArray = numpy_support.numpy_to_vtk(data_portion.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

            imdata.SetDimensions([128, 128, 128])
            # fill the vtk image data object
            imdata.SetSpacing([1, 1, 1])
            imdata.SetOrigin([0, 0, 0])
            imdata.GetPointData().SetScalars(depthArray)

            writer = vtk.vtkMetaImageWriter()
            writer.SetFileName('J:/Program/CT_VSGAN_data/gen_volume/' + name + '/epoch_' + str(idx) + '_fake_V_' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.mha')
            writer.SetInputData(imdata)
            writer.Write()

        '''
        data_portion = data
        # data = data / 255.
        imdata = vtk.vtkImageData()

        depthArray = numpy_support.numpy_to_vtk(data_portion.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

        imdata.SetDimensions([64, 64, 64])
        # fill the vtk image data object
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName(
            'J:/Program/CT_VSGAN_data/gen_volume/' + name + '/epoch_' + str(idx) + '_fake_EC_' + str(i).zfill(2) + '.mha')
        writer.SetInputData(imdata)
        writer.Write()
        '''

    print(str(idx) + ' Finished')

'''
for i in range (0, 16):
    data = np.load('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230331_1_hagan_1e_4_4e_4_1e_4/x_rand_nifti.nii' + str(i) + '.npy')

    data = data[:, :, :]
    # data = data / 255.
    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([128, 128, 128])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230331_1_hagan_1e_4_4e_4_1e_4/x_rand_nifti.nii' + str(i) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()
    '''
'''
name = "real_data_orig_seg_clean"

for i in range (1, 10):
    data = np.load('J:/Program/CT_VSGAN_data/gen_volume/' + name + '/p_' + str(i).zfill(3) + '.npy')
    # data = data / 255.

    # print(data.min())
    # print(data.max())

    data = data[:, :, :]
    data = data / 255.
    imdata = vtk.vtkImageData()

    depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

    imdata.SetDimensions([128, 128, 128])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName('J:/Program/CT_VSGAN_data/gen_volume/' + name + '/p_' + str(i).zfill(3) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()
    '''
'''
volume_path = 'J:/DK_Data_Process/Paper/volume/f_0000008/model.binvox'
if os.path.exists(volume_path):
    with open(volume_path, 'rb') as f:
        volume = binvox_rw.read_as_3d_array(f)
        data = volume.data

        data = data[:, :, :]
        imdata = vtk.vtkImageData()

        depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)

        imdata.SetDimensions([128, 128, 128])
        # fill the vtk image data object
        imdata.SetSpacing([1, 1, 1])
        imdata.SetOrigin([0, 0, 0])
        imdata.GetPointData().SetScalars(depthArray)

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName('J:/DK_Data_Process/Paper/volume/f_0000008/model.mha')
        writer.SetInputData(imdata)
        writer.Write()
        '''
'''
for i in range (0, 1):
    # data = np.load('J:/Program/CT_VSGAN_data/gen_volume/real_data_orig_seg_clean/p_' + str(i).zfill(3) + '.npy')
    # data = np.load('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230423_hagan_1e_4_4e_4_1e_4_orig_seg_num_32/hagan_volume_' + str(i).zfill(2) + '.npy')
    # data = np.load('J:/Program/3D-GAN-pytorch-master/imgs/230425_1_3dgan_log_lr_G_2_5e_3_D_1e_5_orig_seg_clean_num_32_ups/020000_' + str(i).zfill(2) + '.mat.npy')
    data = np.load('J:/Program/3dbraingen-master/output/result/230423_1_braingen_alpha_lr_1e_4_epoch_10K_orig_result_num_32_norm/gen_volume_' + str(i).zfill(2) + '.npy')
    # data = np.load('G:/Datasets/CT-ORG/OrganSegmentations/_spine/result/volume/p_011.npy')

    data = (data - data.min()) / (data.max() - data.min())

    # print(data.min())
    # print(data.max())

    data = data[:, :, :]
    # data = data / 255.

    # data = zoom(data, (2, 2, 2))
    data = np.clip(data, 0.0, 1.0)

    print(data.shape)
    print(data.min())
    print(data.max())

    # np.save('J:/Program/CT_VSGAN_data/gen_volume/real_data_orig_seg_clean_64/p_' + str(i).zfill(3) + '.npy', data)
    # np.save('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230425_hagan_1e_4_4e_4_1e_4_orig_seg_clean_norm/hagan_volume_' + str(i).zfill(2) + '.npy', data)
    np.save('J:/Program/3dbraingen-master/output/result/230423_1_braingen_alpha_lr_1e_4_epoch_10K_orig_result_num_32_norm/gen_volume_' + str(i).zfill(2) + '.npy', data)
    # np.save('J:/Program/3D-GAN-pytorch-master/imgs/230425_1_3dgan_log_lr_G_2_5e_3_D_1e_5_orig_seg_clean_num_32_ups/020000_' + str(i).zfill(2) + '.mat.npy', data)


    imdata = vtk.vtkImageData()
    
    depthArray = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    
    imdata.SetDimensions([128, 128, 128])
    # fill the vtk image data object
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(depthArray)
    
    writer = vtk.vtkMetaImageWriter()
    # writer.SetFileName('J:/Program/CT_VSGAN_data/gen_volume/230422_4_log_E_log_vol_E_no_norm_lr_2e_6_G_no_sim_w_1_D_recon_no_IoU_cont_epoch_410_WGAN_GP_3_RMSprop_code_noise_random_batch_size_4_orig_seg/epoch_400_fake_V_' + str(i).zfill(2) + '.mha')
    # writer.SetFileName('J:/Program/HA-GAN-master/HA-GAN-master/output/imgs/230423_hagan_1e_4_4e_4_1e_4_orig_seg_num_32/hagan_volume_' + str(i).zfill(2) + '.mha')
    # writer.SetFileName('J:/Program/3D-GAN-pytorch-master/imgs/230425_1_3dgan_log_lr_G_2_5e_3_D_1e_5_orig_seg_clean_num_32_ups/020000_' + str(i).zfill(2) + '.mat.mha')
    writer.SetFileName('J:/Program/3dbraingen-master/output/result/230423_1_braingen_alpha_lr_1e_4_epoch_10K_orig_result_num_32_norm/gen_volume_' + str(i).zfill(2) + '.mha')
    # writer.SetFileName('G:/Datasets/CT-ORG/OrganSegmentations/_spine/result/volume/p_011.mha')
    writer.SetInputData(imdata)
    writer.Write()
    '''