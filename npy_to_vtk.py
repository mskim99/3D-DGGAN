import vtk
import numpy as np
from vtk.util import numpy_support
import os
import binvox_rw

name = "230103_7_log_loss_vol_slab_slice_valid_fake_vec_vol_dec_E_lr_2e_6_G_lr_2e_5_recon_no_IoU_cont_remove_layer_D_lr_2e_5_clamp_decay_2_50_epoch_210"

for idx in range(20, 220, 20):
    for i in range (9, 10):
        data = np.load('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_' + str(idx) + '_fake_V_' + str(i).zfill(2) + '.npy')

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
        writer.SetFileName('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_' + str(idx) + '_fake_V_' + str(i).zfill(2) + '.mha')
        writer.SetInputData(imdata)
        writer.Write()

        print(str(idx) + ' Finished')
'''
for i in range (0, 18):
    data = np.load('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_400_fake_V_' + str(i).zfill(2) + '.npy')

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
    writer.SetFileName('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_400_fake_V_' + str(i).zfill(2) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()

for i in range (0, 18):
    data = np.load('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_600_fake_V_' + str(i).zfill(2) + '.npy')

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
    writer.SetFileName('J:/Program/CT_VSGAN/gen_volume/' + name + '/epoch_600_fake_V_' + str(i).zfill(2) + '.mha')
    writer.SetInputData(imdata)
    writer.Write()
    '''

'''
volume_path = './data/orig/train/avg_volume_train.binvox'
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
        writer.SetFileName('./data/orig/train/avg_volume_train.mha')
        writer.SetInputData(imdata)
        writer.Write()
        '''
'''
data = np.load('./test_real.npy')

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
writer.SetFileName('./test_real.mha')
writer.SetInputData(imdata)
writer.Write()
'''