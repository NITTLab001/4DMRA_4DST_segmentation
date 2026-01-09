# before running inference, prepare the 4DMRI volume by generating subvolumes of 3 adjacent slices
# name each subvolume as 'ID#_###_3sl_4DST_input.nii' where ### is the middle slice number starting from 2
# download the pretrained weights 'model_checkpoint_4dST_prnorm_resdense_test.weights.h5' from latest GitHub release

import os
import scipy.io as sio
import nibabel as nib
import numpy as np
from Model_4DST_full import unet_4dst_resdense_prenorm
from losses import dice_coefficient_msl, bce_msl, bce_dice_loss_msl, dice_loss_msl

#%% function to generate data paths
def get_data_paths_prefix(data_dir, data_prefix, data_suffix='input.nii'):
  data_paths = []

  for file in os.listdir(data_dir):
    if file.endswith(data_suffix) and file.startswith(data_prefix):
      data_path = os.path.join(data_dir, file)
      if os.path.exists(data_path):
          data_paths.append(data_path)
  return data_paths

#%% set paths
checkpoint_filepath = 'your path to model_checkpoint_4dST_prnorm_resdense_test.weights.h5 file' 
data_dir = 'folder path were input files are stored' # add / at end 
data_dir_save = 'folder path were inference results are saved'  # add / at end 

#%% model and weight loading
input_shape = (192, 192, 24, 3, 1)  # Example input shape for 4D ASL MRI data
num_channels = 3  # Number of input channels
num_classes = 2  # Number of output classes
kernel_initializer = 'glorot_uniform' 

model = unet_4dst_resdense_prenorm(input_shape, num_channels, num_classes, kernel_initializer)
model.load_weights(checkpoint_filepath)

#%% Inference 
sid = ['YourID#'] # Patient ID
data_paths = get_data_paths_prefix(data_dir, data_prefix=sid, data_suffix='input.nii') # generate data paths based on prefix (ID) and suffic (input)
matr = np.empty((192, 192, 30))  # inference output volume

for x in range(len(data_paths)):  # loops through slices
    slicen = int(data_paths[x][-22:-19])  # slice number, adjust to fine names
    current_input_path = data_paths[x]
    inputdat = nib.load(current_input_path).get_fdata()
    inputdat_batch = inputdat[np.newaxis, ...]  # add batch axis
    prediction = model.predict(inputdat_batch, verbose=0)
    newp = prediction.squeeze()
    matr[:, :, slicen-2] = np.array(newp) # concatenate slices to get volume

#%% to save as .nii
affine = np.eye(4)
nifti_img = nib.Nifti1Image(matr, affine)
fname = '_4DST_pred.nii'  # Name of csv to save
nib.save(nifti_img, data_dir_save + fname)

#%% to save as .mat
fname_mat = '_4DST_pred.mat'  # Name of csv to save
save_dat = {'pred': matr}
sio.savemat(data_dir_save + fname_mat, save_dat)
