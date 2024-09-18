import os
import cv2
import numpy as np
import pandas as pd
fold_df = pd.read_csv('/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/dl-mri/preprocessing/164case.csv')
# case_id
case_ids = fold_df['id'].tolist()
data_dir = '/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/dl-mri/output/'
id_list = case_ids
img_size = 224
outline = 10
# def save_images_as_png(data_dir, id_list, img_size=224, outline=outline):
mask_dir = data_dir + '/mask'
ori_dir = data_dir + '/ori'
output_dir = data_dir + '/png_images_224_10'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for id in id_list:
    for suffix in ['Ap', 'PVP', 'DP']:
        X_file_name = f"{id}_{suffix}.npy"
        m_file_name = f"{id}_{suffix}.npy"

        if not (os.path.exists(os.path.join(ori_dir, X_file_name)) and
                os.path.exists(os.path.join(mask_dir, m_file_name))):
            continue

        X = np.load(os.path.join(ori_dir, X_file_name))
        m = np.load(os.path.join(mask_dir, m_file_name))

                # ...
        # print("Image shape:", X.shape)
        # print("Image dtype:", X.dtype)

        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
         # RGB，
        if len(X.shape) == 3 and X.shape[2] == 3:  # 3，BGR
            X_gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        elif len(X.shape) == 2:  # ，
            X_gray = X
        else:
            raise ValueError("Unsupported image shape: {}".format(X.shape))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # print("Image shape:", X.shape)
        # print("Image dtype:", X.dtype)
        X = clahe.apply(X_gray)

        X = (X - np.min(X)) / (np.max(X) - np.min(X))

        x, y = np.where(m > 0)

        w0, h0 = m.shape
        x_min = max(0, int(np.min(x) - outline))
        x_max = min(w0, int(np.max(x) + outline))
        y_min = max(0, int(np.min(y) - outline))
        y_max = min(h0, int(np.max(y) + outline))

        m = m[x_min:x_max, y_min:y_max]
        X = X[x_min:x_max, y_min:y_max]

        X_m_1 = X.copy()
        if X_m_1.shape[0] != img_size or X_m_1.shape[1] != img_size:
            X_m_1 = cv2.resize(X_m_1, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        X_m_1 = (X_m_1 - np.min(X_m_1)) / (np.max(X_m_1) - np.min(X_m_1))
        X_m_1 = np.expand_dims(X_m_1, axis=-1)
        X_m_1 = np.concatenate([X_m_1, X_m_1, X_m_1], axis=-1)

        #  PNG 
        output_file_name = f"{id}_{suffix}.png"
        cv2.imwrite(os.path.join(output_dir, output_file_name), X_m_1 * 255)