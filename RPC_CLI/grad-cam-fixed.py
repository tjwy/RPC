# from models.model_MCATPathways_copy import MCATPathways
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess_UNI import get_wsi_feature
from PIL import Image
import timm
import pandas as pd
import os
import h5py 
# from models.model_SurvPath_latest1 import SurvPath ###
from models.model_SurvPath_no_freeze_RPC import SurvPath ###
import openslide
from skimage.transform import resize  #  resize 
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, \
                            XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import argparse
from sklearn.preprocessing import StandardScaler
# device，'cuda''cpu'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _init_model(args):
    
    print('\nInit Model...', end=' ')

    if args.modality == "survpath":

        model_dict = {'num_classes': args.n_classes}
        model = SurvPath(**model_dict)

    else:
        raise NotImplementedError
        # 
    print(f"Initialized model: {model.__class__.__name__}")
    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    return model

def pil_to_tensor(image, target_size=(224, 224)):
    #  PIL  NumPy 
    img_array = np.array(image, dtype=np.float32)
    
    # ，RGB
    if img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]  # 
        img_array = np.repeat(img_array, repeats=3, axis=2)  # 

    # RGB，
    elif img_array.shape[2] != 3:
        raise ValueError("Input image is not a RGB image with 3 channels.")

    #  skimage  resize 
    img_array = resize(img_array, target_size, anti_aliasing=True)
    
    #  PIL (HWC)  PyTorch (CHW)
    img_array = img_array.transpose((2, 0, 1))
    
    #  NumPy  PyTorch ，
    img_tensor = torch.from_numpy(img_array) / 255.0
    
    return img_tensor

import pandas as pd
numerical_columns = [
        'age', 'NLR',
        'PLT','tumor_dm', 'Gamma-Glutamyltransferase', 
        'Alkaline_Phosphatase',
        'Cirrhosis'
    ]
def get_clinical_data(clinic_data, case_id):

    categorical_columns = [
'major_resection', 
'Poor_differentiation','MVI', 'Satellite_nodules',
'BCLC',
'AFP400', 
    ]
    # 
    numerical_columns = [
        'age', 'NLR',
        'PLT','tumor_dm', 'Gamma-Glutamyltransferase', 
        'Alkaline_Phosphatase',
        'Cirrhosis'
    ]
    # 
    numerical_data = clinic_data.loc[case_id, numerical_columns]
    
    # z-score
    # numerical_data = (numerical_data - numerical_data.mean()) / numerical_data.std()

    # one-hot
    one_hot_encoded = []
    for column_name in categorical_columns:
        value = clinic_data.loc[case_id, column_name]
        one_hot_encoded.extend([1, 0] if value == 0 else [0, 1])
    
    # one-hot
    clinical_data_list = numerical_data.tolist() + one_hot_encoded

    return clinical_data_list


# ************* GradCAM-customized model definition for multiple input tensors ************* #
class CustomizedModule(torch.nn.Module):
    def __init__(self, model, input_img1, input_img2, input_img3, input_wsi, input_clinical):
        super(CustomizedModule, self).__init__()
        self.model = model
        self.input_img1 = input_img1
        self.input_img2 = input_img2
        self.input_img3 = input_img3
        self.input_wsi = input_wsi
        self.input_clinical = input_clinical
        self.input_image_id = 0

    def set_input_image_id(self, image_id):
        assert image_id in [0, 1, 2], "Invalid image ID"
        self.input_image_id = image_id
    
    def forward(self, x):
        # input_image_id indicates that x should substitute input image tensor
        if self.input_image_id == 0:
            # ，，
            # ，
            x = x  #  x  (1, 3, 224, 224)
            input_img2 = self.input_img2  # 
            input_img3 = self.input_img3  # 
            print('x.shape',x.shape)
            #  torch.stack 
            img_tensor = torch.stack([x, input_img2, input_img3], dim=0)  #  (3, 1, 3, 224, 224)
            print('img_tensor.shape',img_tensor.shape)
            #  view  reshape  (9, 224, 224)
            # img_tensor_flattened = img_tensor.view(1, -1, 224, 224)  # 
            img_tensor_flattened = img_tensor.view( -1, 224, 224)
            print('img_tensor_flattened.shape',img_tensor_flattened.shape)
            img_tensor_input = img_tensor_flattened.permute(1, 2, 0)
            # return_attn1 = False
            inputs = {}
            inputs['x_img'] = img_tensor_input
            inputs['x_wsi'] = self.input_wsi
            inputs['clinical_data'] = self.input_clinical
            # inputs["return_attn"] = False
        #     inputs = {
        #     'x_img': img_tensor_input,  #  input_tensor 
        #     'x_wsi': self.input_wsi,
        #     'clinical_data': self.input_clinical,
        #     'return_attn': return_attn1
        # }
            print('self.model(**inputs)',self.model(**inputs))
            return self.model(**inputs)
        elif self.input_image_id == 1:
            x = x  #  x  (1, 3, 224, 224)
            input_img1 = self.input_img1  # 
            input_img3 = self.input_img3  # 

            #  torch.stack 
            img_tensor = torch.stack([input_img1, x, input_img3], dim=0)  #  (3, 1, 3, 224, 224)
            print('img_tensor.shape',img_tensor.shape)

            #  view  reshape  (9, 224, 224)
            # img_tensor_flattened = img_tensor.viewimg_tensor.view(1, -1, 224, 224)  # 
            img_tensor_flattened = img_tensor.view(-1, 224, 224)
            print('img_tensor_flattened.shape',img_tensor_flattened.shape)
            img_tensor_input = img_tensor_flattened.permute(1, 2, 0)
            inputs = {
            'x_img': img_tensor_input,  #  input_tensor 
            'x_wsi': self.input_wsi,
            'clinical_data': self.input_clinical
        }
            return self.model(**inputs)
            # return self.model(self.input_img1, x, self.input_img3, self.input_wsi, self.input_clinical)
        elif self.input_image_id == 2:
            x = x  #  x  img_tensor.view(1, -1, 224, 224)
            input_img1 = self.input_img1  # 
            input_img2 = self.input_img2  # 

            #  torch.stack 
            img_tensor = torch.stack([ input_img1, input_img2,x], dim=0)  #  (3, 1, 3, 224, 224)

            #  view  reshape  (9, 224, 224)
            # img_tensor_flattened = img_tensor.view(-1, 224, 224)  # 
            img_tensor_flattened = img_tensor.view(-1, 224, 224)
            img_tensor_input = img_tensor_flattened.permute(1, 2, 0)
            inputs = {
            'x_img': img_tensor_input,  #  input_tensor 
            'x_wsi': self.input_wsi,
            'clinical_data': self.input_clinical
        }
            return self.model(**inputs)
            # return self.model(self.input_img1, self.input_img2, x, self.input_wsi, self.input_clinical)
        else:
            raise ValueError("Invalid self input image ID")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors        
def create_color_bar_with_ticks(height, cmap_name='jet'):
    """
    01，height。
    """
    # 
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # figureaxes
    fig, ax = plt.subplots(figsize=(0.5, height / 100))  # 0.5，
    ax.set_axis_off()  # 

    # 
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cbar = mcolors.Colormap(norm, cmap_name)
    orientation = 'vertical'
    color_bar = plt.colorbar(mcolors.ScalarMappable(norm=norm, cmap=cbar), cax=ax, orientation=orientation)
    
    # 
    color_bar.set_ticks([0, 1])
    color_bar.set_ticklabels(['0', '1'])

    # 
    plt.savefig(output_colorbar_path1, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--modality', type=str, default='survpath', help='modality')
    argparser.add_argument('--model_path', type=str, default='checkpoints/model_best.pt', help='model path')
    argparser.add_argument('--n_classes', type=int, default=2, help='num_classes')
    args = argparser.parse_args()
    argparser.add_argument('--return_attn', type=bool , default='False')
    model = _init_model(args)
    # load checkp
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print('model loaded!')

    df_cases = pd.read_csv('/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/SurvPath/datasets_csv/metadata/demo_tcga_lihc.csv')
    clinic_data_list = pd.read_csv('/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/SurvPath/datasets_csv/clinical_data/tcga_lihc_clinical_579.csv', index_col=1)
    
    # print(clinic_data_list.columns.tolist())
    scaler = StandardScaler()
    trian_ids = clinic_data_list.index[clinic_data_list[f'fold-2'] == 'training'].tolist()
    print('train_ids_len',len(trian_ids))
    raw_train_data = clinic_data_list[clinic_data_list.index.isin(trian_ids)]
    scaler.fit(raw_train_data[numerical_columns])
    clinic_data_list[numerical_columns] = pd.DataFrame(scaler.transform(clinic_data_list[numerical_columns]),columns=numerical_columns, index=clinic_data_list.index)
    
    # case
    for index, row in df_cases.iterrows():
        case_id = row['case_id']
        slide_id = row['slide_id']

        ori_dir = '/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/dl-mri/output'
        output_path = '/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/SurvPath/MR_heatmap/MR_heatmap9_4_final/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img1 = cv2.imread(os.path.join(ori_dir,f'png_images_224_10', f"{case_id}_Ap.png"))
        img2 = cv2.imread(os.path.join(ori_dir,f'png_images_224_10', f"{case_id}_PVP.png"))
        img3 = cv2.imread(os.path.join(ori_dir,f'png_images_224_10',f"{case_id}_DP.png"))

        cv2.imwrite(os.path.join(output_path, f'{case_id}_1.jpg'), img1)
        cv2.imwrite(os.path.join(output_path, f'{case_id}_2.jpg'), img2)
        cv2.imwrite(os.path.join(output_path, f'{case_id}_3.jpg'), img3)

        # WSI = openslide.OpenSlide(f'/hpc2hdd/home/yangwu/WY/svs_rename_10-30/HE/{slide_id}.svs')
        wsi_feature = []
        # Modify the code to load HDF5 files

        wsi_path = os.path.join('/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/im4MEC/feature_bags_UNI_256um', '{}_features.h5'.format(slide_id.rstrip('.svs')))
        with h5py.File(wsi_path, 'r') as hf:
            wsi_bag = torch.tensor(np.array(hf['features']))  # Assuming that the features are stored under 'features' key
            wsi_feature.append(wsi_bag)
        wsi_feature = torch.cat(wsi_feature, dim=0)

        img1_tensor = pil_to_tensor(img1).to(device).unsqueeze(0)
        img2_tensor = pil_to_tensor(img2).to(device).unsqueeze(0)
        img3_tensor = pil_to_tensor(img3).to(device).unsqueeze(0)
        print('img1_tensor.shape',img1_tensor.shape)
        # case_idslide_id 
        cli_data = get_clinical_data(clinic_data_list, case_id)
        # cli_data，NumPy
        cli_data_np = np.array(cli_data, dtype=np.float32)
        clinical = torch.from_numpy(cli_data_np).to(device)
        print('clinical',clinical.shape)
        print('clinical',clinical)
        # # 
        # wsi_feature = get_wsi_feature(device, WSI, 224 ,512)

        wsi_feature = wsi_feature.to(device).unsqueeze(0)
        print('wsi_feature',wsi_feature.shape)

        # ************* GradCAM critical codes ************* #
    ### 
        wrapper_model = CustomizedModule(model, img1_tensor, img2_tensor, img3_tensor, wsi_feature, clinical)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        cam = GradCAM(model=wrapper_model,
                target_layers=[wrapper_model.model.seq_vit_model.vit_base_model.layer4[-1].conv3])
                # target_layers=[wrapper_model.model.seq_vit_model.vit_base_model.layer4[-1]])

        print('cam',cam)
        wrapper_model.set_input_image_id(0)
        grayscale_cam1 = cam(input_tensor=img1_tensor)

        print('grayscale_cam1.shape',grayscale_cam1.shape)
        grayscale_cam1 = grayscale_cam1[0, :]
        print('grayscale_cam1.shape',grayscale_cam1.shape)
        print('grayscale_cam1',grayscale_cam1)

            #  np.float32 
        img1 = img1.astype(np.float32)

        #  [0, 255] ， [0, 1]
        if img1.max() > 1.0:
            img1 /= 255.0

        #  [0, 1] 
        img1 = np.clip(img1, 0, 1)
        # print('img1',img1)
        image_weight = 0.6
        visualization1 = show_cam_on_image(img1, grayscale_cam1,use_rgb= False, image_weight=image_weight)
        # output_image_path1 = f'{case_id}_grad_cam1.jpg'
        output_image_path1 = os.path.join(output_path, f'{case_id}_grad_cam1.jpg')
        cv2.imwrite(output_image_path1, visualization1)


        print('Grad-CAM image saved at', output_image_path1)

        wrapper_model.set_input_image_id(1)
        # grayscale_cam2 = cam(input_tensor=img2_tensor, target_category=None)
        grayscale_cam2 = cam(input_tensor=img2_tensor)

        grayscale_cam2 = grayscale_cam2[0, :]
        print('grayscale_cam2',grayscale_cam2)
                #  np.float32 
        img2 = img2.astype(np.float32)

        #  [0, 255] ， [0, 1]
        if img2.max() > 1.0:
            img2 /= 255.0

        #  [0, 1] 
        img2 = np.clip(img2, 0, 1)

        visualization2 = show_cam_on_image(img2, grayscale_cam2,use_rgb= False, image_weight=image_weight)
        # output_image_path2 = f'{case_id}_grad_cam2.jpg'
        output_image_path2 = os.path.join(output_path, f'{case_id}_grad_cam2.jpg')
        cv2.imwrite(output_image_path2, visualization2)
        print('Grad-CAM image saved at', output_image_path2)


        wrapper_model.set_input_image_id(2)
        grayscale_cam3 = cam(input_tensor=img3_tensor)

                    #  np.float32 
        img3 = img3.astype(np.float32)

        #  [0, 255] ， [0, 1]
        if img3.max() > 1.0:
            img3 /= 255.0

        #  [0, 1] 
        img3 = np.clip(img3, 0, 1)

        grayscale_cam3 = grayscale_cam3[0, :]
        visualization3 = show_cam_on_image(img3, grayscale_cam3,use_rgb= False, image_weight=image_weight)
        # output_image_path3 = f'{case_id}_grad_cam3.jpg'
        output_image_path3 = os.path.join(output_path, f'{case_id}_grad_cam3.jpg')
        cv2.imwrite(output_image_path3, visualization3)
        
        print('Grad-CAM image saved at', output_image_path3)
        print('All Grad-CAM images saved!')
        
        colormap = cv2.COLORMAP_JET

        # 
        output_heatmap_path1 = os.path.join(output_path, f'{case_id}_heatmap1.jpg')
        output_heatmap_path2 = os.path.join(output_path, f'{case_id}_heatmap2.jpg')
        output_heatmap_path3 = os.path.join(output_path, f'{case_id}_heatmap3.jpg')

        #  [0, 1] ， [0, 255]  np.uint8 
        grayscale_cam1 = (grayscale_cam1 * 255).astype(np.uint8)
        grayscale_cam2 = (grayscale_cam2 * 255).astype(np.uint8)
        grayscale_cam3 = (grayscale_cam3 * 255).astype(np.uint8)

        # 
        heatmap1 = cv2.applyColorMap(grayscale_cam1, colormap)
        heatmap2 = cv2.applyColorMap(grayscale_cam2, colormap)
        heatmap3 = cv2.applyColorMap(grayscale_cam3, colormap)

        # 
        cv2.imwrite(output_heatmap_path1, heatmap1)
        cv2.imwrite(output_heatmap_path2, heatmap2)
        cv2.imwrite(output_heatmap_path3, heatmap3)

        # 
        fig, ax = plt.subplots()
        # 
        ax.imshow(visualization3, cmap='jet')
        
        # 
        ax.axis('off')

        # visualization3
        norm = plt.Normalize(vmin=np.min(visualization3), vmax=np.max(visualization3))

        # 
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=norm), ax=ax)
        # 
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['0', '1'])
        # 
        cbar.set_label('Attention Weight')

        # 
        output_image_path3_with_colorbar = os.path.join(output_path, f'{case_id}_grad_cam3_with_colorbar.jpg')
        plt.savefig(output_image_path3_with_colorbar, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print('Grad-CAM image with color bar saved at', output_image_path3_with_colorbar)
