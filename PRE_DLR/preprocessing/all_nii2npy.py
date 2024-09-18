import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted

# 
if not os.path.exists('output_all'):
    os.makedirs('output_all')
if not os.path.exists(os.path.join('output_all', 'mask')):
    os.makedirs(os.path.join('output_all', 'mask'))
if not os.path.exists(os.path.join('output_all', 'ori')):
    os.makedirs(os.path.join('output_all', 'ori'))

clini_df_path = './clini.csv'
clini_df = pd.read_csv(clini_df_path)

def process_and_save_masked_layers(image_path, mask_path, id_prefix, save_dir='mask', is_mask=True):
    mask_image = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)

    all_layers = []

    for layer_idx in range(mask_array.shape[0]):
        if np.any(mask_array[layer_idx, :, :] > 0):  
            if is_mask:
                img_to_save = mask_array[layer_idx, :, :]
            else:
                img_to_save = image_array[layer_idx, :, :]

            all_layers.append(img_to_save)

    # 
    stacked_layers = np.stack(all_layers, axis=0)
    file_name = f'{id}_{id_prefix}.npy'
    np.save(os.path.join('output_all', save_dir, file_name), stacked_layers)

for index, row in clini_df.iterrows():
    id = row['id']
    base_dir = row['ori_path']

    ap_ori_path = os.path.join(base_dir, 'AP.nii')
    ap_mask_path = os.path.join(base_dir, 'AProi.nii')

    process_and_save_masked_layers(ap_ori_path, ap_mask_path, 'Ap', 'ori')
    process_and_save_masked_layers(ap_mask_path, ap_mask_path, 'Ap', 'mask')

    pvp_ori_path = os.path.join(base_dir, 'PVP.nii')
    pvp_mask_path = os.path.join(base_dir, 'PVProi.nii')

    process_and_save_masked_layers(pvp_ori_path, pvp_mask_path, 'PVP', 'ori')
    process_and_save_masked_layers(pvp_mask_path, pvp_mask_path, 'PVP', 'mask')

    dp_ori_path = os.path.join(base_dir, 'DP.nii')
    dp_mask_path = os.path.join(base_dir, 'DProi.nii')

    process_and_save_masked_layers(dp_ori_path, dp_mask_path, 'DP', 'ori')
    process_and_save_masked_layers(dp_mask_path, dp_mask_path, 'DP', 'mask')