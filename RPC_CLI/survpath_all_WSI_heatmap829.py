import openslide
from shapely.affinity import scale
from shapely.geometry import box
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from preprocess_UNI import get_wsi_feature
from PIL import Image
import timm
import pandas as pd
import os
import h5py 
from models.model_SurvPath_freeze_RPC import SurvPath ###
from utils.core_utils_latest1 import _init_model
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

from preprocess_UNI import (
    create_tissue_mask,
    create_tissue_tiles,
    extract_features,
    # load_encoder,
)
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont  #  ImageFont
def get_wsi_properties(wsi):
    # """
    # # WSI，（）。
    # """
   properties = wsi.properties
   mpp_x = float(properties.get('openslide.mpp-x', 0))  # （）
   mpp_y = float(properties.get('openslide.mpp-y', 0))  # （）
   return mpp_x, mpp_y

def draw_scale_bar(image, length_in_pixels, length_in_units, position, units="µm", font_size=12):
    """
    。
    """
    draw = ImageDraw.Draw(image)
    # 
    draw.line([position, (position[0] + length_in_pixels, position[1])], fill="black", width=2)
    # 
    text = f"{length_in_units:.2f} {units}"
    font = ImageFont.truetype("arial.ttf", font_size)
    textbox = draw.textbbox(position, text, font=font)
    text_position = (position[0] + length_in_pixels + 5, position[1] - (textbox[3] - textbox[1]) // 2)
    draw.text(text_position, text, fill="black", font=font)
    return image
def gen_dataset_multi_images(data_dir, cid, img_size=224, outline=10):
    mask_dir = data_dir + '/mask'
    ori_dir = data_dir + '/ori'

    stacked_image = None  # ID
    for suffix in ['Ap', 'PVP', 'DP']:
        X_file_name = f"{cid}_{suffix}.npy"
        m_file_name = f"{cid}_{suffix}.npy"

        if not (os.path.exists(os.path.join(ori_dir, X_file_name)) and
                os.path.exists(os.path.join(mask_dir, m_file_name))):
            continue

        X = np.load(os.path.join(ori_dir, X_file_name))
        m = np.load(os.path.join(mask_dir, m_file_name))

        # ...
        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        X = clahe.apply(X)
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

        if stacked_image is None:
            stacked_image = X_m_1
        else:
            stacked_image = np.concatenate([stacked_image, X_m_1], axis=-1)
    return stacked_image

def predict_attention_matrix_rad3(model, input):
    with torch.no_grad():
        logits, attn_pathways, cross_attn_pathways, cross_attn_histology = model(**input)
        # print("Attention matrix shape:", cross_attn_pathways.shape)
        # print("Cross Attention matrix shape:", cross_attn_histology.shape)
    return logits, attn_pathways.cpu().numpy(), cross_attn_pathways.cpu().numpy(), cross_attn_histology.cpu().numpy()

def get_display_image(wsi, display_level):
    # just take the last top level of the slide to display the attention heatmap on
    assert display_level < (len(wsi.level_dimensions) - 1)
    display_image = wsi.read_region(
        (0, 0), display_level, wsi.level_dimensions[display_level]
    )

    # Determine the scale factor to scale the tile coordinates down to the desired heatmap resolution
    scale_factor = 1 / wsi.level_downsamples[display_level]
    return display_image, scale_factor


def standardize_scores(raw):
    # Note that the Z-scores only take the attention distribution of this slide into account.
    # This shouldn't matter for interpretation though, as classification is ultimately performed on the top-K attended tiles.
    # This makes the absolute attention value of a tile pretty much meaningless.
    z_scores = (raw - np.mean(raw)) / np.std(raw)
    z_scores_s = z_scores + np.abs(np.min(z_scores))
    z_scores_s /= np.max(z_scores_s)
    return z_scores_s

def scale_rectangles(raw_rect_bounds, scale_factor):
    rects = []
    for coords in raw_rect_bounds:
        # reconstruct the rectangles from the bounds using Shapely's box utility function
        minx, miny, maxx, maxy = coords
        rect = box(minx, miny, maxx, maxy)

        # scale the rectangles using the scale factor
        rect_scaled = scale(
            rect, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
        )
        rects.append(rect_scaled)
    return rects

def build_scoremap(src_img, rect_shapes, scores):
    # Note: We assume the rectangles do not overlap!

    # Create an empty array with the same dimensions as the image to hold the attention scores.
    # Note that the dimensions of the numpy array-based representation of the Image are ordered differently than when using Image.size()
    h, w, _ = np.asarray(src_img).shape
    score_map = np.zeros(dtype=np.float32, shape=(h, w))

    # Assign the scores to the buffer for each rectangle.
    for rect, score in zip(rect_shapes, scores):
        minx, miny, maxx, maxy = rect.bounds

        # Note that we round the rectangle coordinates, as they have turned into floats
        # after scaling.
        score_map[round(miny) : round(maxy), round(minx) : round(maxx)] = score

    return score_map


def scoremap_to_heatmap(score_map):
    # Build a false-color map
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * score_map), cv2.COLORMAP_JET)

    # OpenCV works in BGR, so we'll need to convert the result back to RGB first for Image to understand it.
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGBA)
    assert heatmap.dtype == np.dtype("uint8")

    # Adjust the overlay opacity (0 = completely transparent)
    heatmap[..., 3] = 60

    # The jet heatmap sets all 0 scores to [0,0,128,255] (blue). This will make the background blue. We don't want that.
    # Set these pixels to be white and transparent instead.
    heatmap[np.where(score_map == 0)] = (255, 255, 255, 0)

    assert heatmap.dtype == np.dtype("uint8")
    assert heatmap.shape[2] == 4
    return Image.fromarray(heatmap, mode="RGBA")

def load_trained_model(
    device,
    checkpoint_path,
    # model_size,
    # input_feature_size,
    n_classes,
):
    model = SurvPath(
        wsi_embedding_dim=1024,
        # img_embedding_dim=768,  # ：img768
        # dropout=0.15,
        num_classes=2,###4
        wsi_projection_dim=256,
        # cli_embedding_dim=17,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def get_features(model, device, wsi, tiles, workers, out_size, batch_size):
    generator = extract_features(
        model,
        device,
        wsi,
        tiles,
        workers=workers,
        out_size=out_size,
        batch_size=batch_size,
    )

    feature_bag = []
    for predicted_batch in generator:
        feature_bag.append(predicted_batch)

    features = torch.from_numpy(np.vstack([f for f, c in feature_bag]))
    coords = np.vstack([c for _, c in feature_bag])
    return features, coords

def get_class_names(manifest):
    df = pd.read_csv(manifest)
    n_classes = len(df["class_label"].unique())
    class_names = {}
    for i in df["class_label"].unique():
        name = df[df["class_label"] == i]["class"].unique()[0]
        class_names[i] = name
    assert len(class_names) == n_classes
    return class_names

numerical_columns = [
        'age', 'NLR',
        'PLT','tumor_dm', 'Gamma-Glutamyltransferase', 
        'Alkaline_Phosphatase',
        'Cirrhosis'
    ]
def get_clinical_data(clinic_data, case_id):
    # 
#     categorical_columns = [
#         'major_resection',  'tumor_size(5cm)', 'Differentiation', 'MVI', 'Satellite_nodules',
#         'BCLC', 'AFP400', 'tumor_num(01)'
#     ]

#     # 
#     numerical_columns = [
# 'age', 'WBC', 'Lymphocytes(%)', 'RBC', 'PLT', 
#         'Alkaline_Phosphatase', 'tumor_dm', 'Gamma-Glutamyltransferase', 'BUN', 'AFP',
#     ]
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# ，
def save_high_score_patches(wsi_object, coords, scores, save_dir, top_k, target_size=1024, score_type='top'):
    # （'top'  'bottom'）
    if score_type == 'top':
        sorted_indices = np.argsort(scores)[::-1]  # 
    elif score_type == 'bottom':
        sorted_indices = np.argsort(scores)  # 
    else:
        raise ValueError("score_type must be 'top' or 'bottom'")
    os.makedirs(save_dir, exist_ok=True)
    #  Top-k  Bottom-k 
    top_k_indices = sorted_indices[:top_k]

    for idx in top_k_indices:
        coord = coords[idx]
        score = scores[idx]
        # 
        patch = wsi_object.read_region(
            (int(coord[0]), int(coord[1])),
            0,
            (int(coord[2]) - int(coord[0]), int(coord[3]) - int(coord[1])),
        ).convert('RGB')
        
        # 
        patch_resized = patch.resize((target_size, target_size), Image.LANCZOS)
        
        # 
        score_str = f"{score:.2f}" if score_type == 'top' else f"{-score:.2f}"  # 
        file_name = f"patch_{coord[0]}_{coord[1]}_{score_type}_{score_str}.png"
        patch_resized.save(os.path.join(save_dir, file_name))

    print(f"Finished saving {score_type} {top_k} patches resized to {target_size} pixels each.")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.patch_save_dir, exist_ok=True)
    df_cases = pd.read_csv(args.input_csv)
    clinic_data_list = pd.read_csv(args.path_to_clinic_data, index_col=1)

    scaler = StandardScaler()
    trian_ids = clinic_data_list.index[clinic_data_list[f'fold-2'] == 'training'].tolist()
    print('train_ids_len',len(trian_ids))
    raw_train_data = clinic_data_list[clinic_data_list.index.isin(trian_ids)]
    scaler.fit(raw_train_data[numerical_columns])
    clinic_data_list[numerical_columns] = pd.DataFrame(scaler.transform(clinic_data_list[numerical_columns]),columns=numerical_columns, index=clinic_data_list.index)
        
        # case
    for index, row in df_cases.iterrows():

        slide_id = row['slide_id']
        case_id = slide_id[:10]
        print(f"Predicting attention map for {slide_id}")
        try:
            WSI = openslide.open_slide(f'/hpc2hdd/JH_DATA/share/fhuang743/yangwu_yangwu_ALL_projects/OLD/svs_rename_10-30_save/HE/{slide_id}.svs')
            mpp_x, mpp_y = get_wsi_properties(WSI)
            # slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))
            # case_id = slide_id[:10]

            print('slide_id',slide_id)
            print('case_id',case_id)

            img_data = gen_dataset_multi_images(args.rad_data_dir, case_id, img_size=224, outline=10)

            img_data = torch.from_numpy(img_data).to(device)
            img_features = img_data### cross——attention

            
            cli_data = get_clinical_data(clinic_data_list, case_id)
            # cli_data，NumPy
            cli_data_np = np.array(cli_data, dtype=np.float32)
            cli_data = torch.from_numpy(cli_data_np).to(device)

            class_names = get_class_names(args.manifest)
            n_classes = len(class_names)

            local_dir = "/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"

            feature_extractor_model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
            feature_extractor_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
            feature_extractor_model.eval()
            feature_extractor_model.to(device)### ?

            attn_model = load_trained_model(
                device, 
                args.attn_checkpoint,
                n_classes,
            )
            attn_model.to(device)

            display_image, scale_factor = get_display_image(WSI, args.display_level)
            scaled_tissue_mask = create_tissue_mask(WSI, WSI.get_best_level_for_downsample(64))

            tiles = create_tissue_tiles(
                WSI, scaled_tissue_mask, args.tile_size, offsets_micron=None
            )

            wsi_features, coords = get_features(
                feature_extractor_model,
                device,
                WSI,
                tiles,
                args.workers,
                args.out_size,
                args.batch_size,
            )
            # print('coords',coords)
            
            wsi_features = wsi_features.to(device).unsqueeze(0)

            # print('cli_data',cli_data)
            # print('wsi_features',wsi_features.shape)
            # print('img_features',img_features.shape)

            input = dict(x_wsi=wsi_features, x_img=img_features, return_attn=True, clinical_data=cli_data)

            ### 
            # A_raw = predict_attention_matrix(attn_model, input)

            logits, attn_pathways, cross_attn_pathways, cross_attn_histology = predict_attention_matrix_rad3(attn_model, input)
            # print('cross_attn_pathways',cross_attn_pathways)
            # print('cross_attn_histology',cross_attn_histology)

            n_slices = 3
            attn_scores = []
            maps_per_slice = []
            for slice_idx in range(n_slices):

                raw_attn = cross_attn_pathways[slice_idx]
                scaled_rects = scale_rectangles(coords, scale_factor)
                # print('scaled_rects',scaled_rects)
                z_scores = standardize_scores(raw_attn)
                print('min z_scores',np.min(z_scores))
                print('max z_scores',np.max(z_scores))
                # print('display_image',display_image)
                scoremap = build_scoremap(display_image, scaled_rects, z_scores)
                attn_scores.append(z_scores)
                maps_per_slice.append(scoremap)

            # Merge the score maps of each offset for each class and save the result.
            # for slice_idx in range(n_slices):
                overlay = scoremap_to_heatmap(scoremap)
                
                # print(display_image)
                # print(overlay)
                result = Image.alpha_composite(display_image, overlay)
                
                outpath = os.path.join(
                    args.output_dir, f"{slide_id}_attn_class_{slice_idx}.jpg"
                )
                print(f"Exporting {outpath}")
                # Note that we discard the alpha channel because JPG does not support transparancy.
                result.convert("RGB").save(outpath)    
            ave_scores = np.mean(attn_scores,axis=0)
            print('min ave_scores',np.min(ave_scores))
            print('max ave_scores',np.max(ave_scores))
            average_map = np.mean(maps_per_slice, axis=0)
            print('min average_map',np.min(average_map))
            print('max average_map',np.max(average_map))
            overlay_3 = scoremap_to_heatmap(average_map)
            result_3 = Image.alpha_composite(display_image, overlay_3)
            outpath_3 = os.path.join(args.output_dir, f"{slide_id}_attn_class_average.jpg")
            result_3.convert("RGB").save(outpath_3)
            # 
            # scale_bar_length_in_microns = 500  # 500
            # scale_bar_length_in_pixels = int(scale_bar_length_in_microns / mpp_x)
            # position = (10, 10)
            # result_3_with_scale = draw_scale_bar(result_3, length_in_pixels=scale_bar_length_in_pixels, length_in_units=scale_bar_length_in_microns, position=position, units="µm")
            # outpath_3 = os.path.join(args.output_dir, f"{slide_id}_attn_class_average.jpg")
            # result_3_with_scale.convert("RGB").save(outpath_3)
            print("Finished.")
            # ， scores  coords ，
            #  Top-5 ，patch_size  224
            top_k = 8
            # patch_size = 224
            # 
            save_high_score_patches(
                wsi_object=WSI,
                coords=coords,
                scores=ave_scores,
                save_dir=os.path.join(args.patch_save_dir,slide_id, 'top'),
                top_k=top_k,
                score_type='top'
            )
            # 
            save_high_score_patches(
                wsi_object=WSI,
                coords=coords,
                scores=ave_scores,
                save_dir=os.path.join(args.patch_save_dir, slide_id,'bottom'),
                top_k=top_k,
                score_type='bottom'
            )
        except Exception as e:
            # ，case
            print(f"Error processing case {case_id}: {e}")
            continue  # ，case

        finally:
            # ，（）
            print("Finished processing this case, moving to next one.")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attention heatmap generation script")

    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to input WSI file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="path to manifest. This is just to retrieve class names and ensure consistency.",
        required=True,
    )
    parser.add_argument(
        "--attn_checkpoint",
        type=str,
        help="Attention model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--path_to_clinic_data",
        type=str,
        help="path_to_clinic_data",
        required=True,
    )
    parser.add_argument(
        "--tile_size",
        help="desired tile size in microns - should be the same as feature extraction model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--rad_data_dir",
        type=str,
        help="rad_data_dir",
        required=True,
    )
    parser.add_argument(
        "--input_feature_size",
        help="The size of the input features from the feature bags.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--out_size",
        help="resize the square tile to this output size (in pixels)",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loader. Only relevant when using a GPU.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--display_level",
        help="Control the resolution of the heatmap by selecting the level of the slide used for the background of the overlay",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--patch_save_dir",
        help="Control the resolution of the heatmap by selecting the level of the slide used for the background of the overlay",
        type=str,
    )
    args = parser.parse_args()
    main(args)

#### attention score topkpatch，，patch，