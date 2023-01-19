import vtk
import numpy as np
from vtk.util import numpy_support
import os
import binvox_rw

name = "230119_4_log_E_no_norm_lr_2e_6_G_recon_no_IoU_cont_D_no_norm_DM_recon_no_IoU_cont_epoch_410_input_rate_c_n_1_1_WGAN_GP_RMSprop_lambda_1e_4"

for idx in range(40, 440, 40):
    for i in range (8, 10):
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