import torch
import numpy as np 
# from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

# from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention_rad import FeedForward, MMAttentionLayer
import pdb

import math
import pandas as pd
import timm
import random
import numpy

# numpy.random.seed(42)
# random.seed(42)
# torch.manual_seed(42)

def exists(val):
    return val is not None

# def SeqVIT:
class SeqVIT(nn.Module):
    def __init__(self,sequence_length=3, image_size=224):
        super(SeqVIT, self).__init__()
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.vit_base_model = timm.create_model(
            # 'vit_base_patch16_224',
            'resnet50',
            pretrained=True,
            num_classes=0,  # Removing the classification head
        )
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        outputs = []
        inputs = inputs.float() ### （224，224，9）
        inputs = inputs.permute(2, 0, 1).unsqueeze(0)
        for t in range(self.sequence_length):
            output_t = self.vit_base_model(inputs[:, t * self.sequence_length:(t + 1) * self.sequence_length:,:, :])
            outputs.append(output_t)
        stacked_outputs = torch.stack(outputs, dim=1)  # Use torch.stack instead of torch.cat
        return stacked_outputs

class SurvPath(nn.Module):
    def __init__(
        self, 
        wsi_embedding_dim=1024,
        img_embedding_dim=2048,  # ：img768
        dropout= 0.1,
        num_classes=2,###4
        wsi_projection_dim=256,
        ###
        image_size=224,
        sequence_length=3
    ):
        super(SurvPath, self).__init__()

        #---> general props
        self.dropout = dropout
        self.num_classes = num_classes
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.img_embedding_dim = img_embedding_dim
        self.num_slices = 3 ##

        #---> wsi preprocessing and projection layer remains the same
        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )
                # ，
        seqvit_weights = torch.load('/hpc2hdd/JH_DATA/share/yangwu/yangwu_yangwu_ALL_projects/623/dl-mri/resnet50_checkpoint_save/checkpoint_fold_3_0.0001_0.001.pt')
        self.seq_vit_model = SeqVIT(sequence_length=3, image_size=224)

        # ，
        self.seq_vit_model.load_state_dict(seqvit_weights, strict=False)
        for param in self.seq_vit_model.parameters():
            param.requires_grad = False
     
        #  SeqVIT  GPU 
        self.seq_vit_model = self.seq_vit_model.to("cuda")

        #---> SeqVITimg_featureencoder
                #---> wsi preprocessing and projection layer remains the same
        self.img_projection_net = nn.Sequential(
            nn.Linear(self.img_embedding_dim, self.wsi_projection_dim),
        )
        #---> cross attention props remain the same
        self.identity = nn.Identity()
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            # num_modalities=3  # ：wsi  ##
        )
        #---> logits props remain the same
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)
        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim , int(self.wsi_projection_dim // 4)),
            # nn.Linear(256+16, 64),
            # nn.Linear(16, 64),
            nn.ReLU(),
            # nn.Dropout(dropout), ### 
            nn.Linear(64, 1),
        )
        ###  w+r no cli
        # self.alpha = nn.Parameter(torch.tensor(1.0))
        # self.clinical_fc = nn.Sequential(
        #     nn.Linear(23, 64),  # 128
        #     nn.ELU(),  ### ELU
        #     nn.Dropout(dropout),
        #     nn.Linear(64, 16)  # 
        # )
    def forward(self, **kwargs):
        wsi_features = kwargs['x_wsi'] 

        img_data = kwargs['x_img']  # 

        img_features = self.seq_vit_model(img_data)

        clinical_features = kwargs['clinical_data'] ###   #### here   
        
        mask = None
        # return_attn = kwargs["return_attn"]
        return_attn = False

        # WSI
        wsi_proj = self.wsi_projection_net(wsi_features)

        # # img
        img_proj = self.img_projection_net(img_features)

        # 
        tokens = torch.cat([img_proj, wsi_proj], dim=1)
        tokens = self.identity(tokens)

        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)
        # print("mm_embed shape:", mm_embed.shape)
        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        # print("mm_embed shape after feedforward and layer norm:", mm_embed.shape)
        #---> aggregate 
        # modality specific mean 
        paths_postSA_embed = mm_embed[:, :self.num_slices, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_slices:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # print("WSI Features shape after passing to model:", wsi_postSA_embed.shape)
                # WSI             
        # if clinical_features is not None:
        #     assert isinstance(clinical_features, torch.Tensor), "Clinical features should be a torch.Tensor."

        #     clinical_proj = clinical_features.to(torch.float).unsqueeze(0)
        #     # print("Clinical Features shape before passing to model:", clinical_proj.shape)
        #     # print("Clinical Features shape before passing to model:", clinical_proj)
        #     clinical_proj = self.clinical_fc(clinical_proj)
        #     clinical_embedding = self.alpha * clinical_proj
        #     embedding = torch.cat([wsi_postSA_embed, paths_postSA_embed,clinical_embedding], dim=1)
        # else:
        #     embedding = torch.cat([wsi_postSA_embed, paths_postSA_embed], dim=1)
            
        # clinical_proj = clinical_features.to(torch.float).unsqueeze(0)
        # clinical_proj = self.clinical_fc(clinical_proj)

        # embedding = clinical_proj
        embedding = torch.cat([wsi_postSA_embed, paths_postSA_embed], dim=1)
        # print("Clinical Features shape after passing to model:", embedding.shape)
        #---> get logits
        logits = self.to_logits(embedding)

        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits
