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
    def __init__(self,sequence_length=3, image_size=224,drop_rate=0.1, pretrained=False):
        super(SeqVIT, self).__init__()
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.drop_rate = drop_rate
        self.vit_base_model = timm.create_model(
            # 'vit_base_patch16_224',
            'resnet50',
            pretrained=pretrained,
            num_classes=0,  # Removing the classification head
            # img_size= image_size,
            # drop_rate= 0.1,   # dropout  0.1
        )
        # self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        outputs = []
        inputs = inputs.float() # ## 2242249B2242249

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        elif inputs.dim() != 4:
            raise ValueError(f"Unexpected MRI tensor shape: {tuple(inputs.shape)}")

        if inputs.shape[-1] != self.sequence_length * 3:
            raise ValueError(
                f"Expected last MRI channel dim to be {self.sequence_length * 3}, got {inputs.shape[-1]}"
            )

        inputs = inputs.permute(0, 3, 1, 2).contiguous()
        # print("Inputs shape:", inputs.shape)
        for t in range(self.sequence_length):
            c_start = t * 3
            c_end = (t + 1) * 3
            output_t = self.vit_base_model(inputs[:, c_start:c_end, :, :])
            outputs.append(output_t)
        stacked_outputs = torch.stack(outputs, dim=1)  # Use torch.stack instead of torch.cat
        # dropout_output = self.dropout(stacked_outputs)
        return stacked_outputs

class SurvPath(nn.Module):
    def __init__(
        self, 
        wsi_embedding_dim=1024,
        img_embedding_dim=2048,  # img768
        dropout= 0.1,
        num_classes=2,# ##4
        wsi_projection_dim=256,
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
        # SeqVIT  SurvPath 
        # self.seq_vit_model = SeqVIT(sequence_length=sequence_length, image_size=image_size, drop_rate=dropout)
        self.seq_vit_model = SeqVIT(sequence_length=sequence_length, image_size=image_size, drop_rate=0.1)        
        # SeqVIT  GPU 
        if torch.cuda.is_available():
            self.seq_vit_model = self.seq_vit_model.to("cuda")

        # ---> SeqVITimg_featureencoder
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
            # num_modalities=3  # wsi  ##
        )
        #---> logits props remain the same
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)
        self.to_logits = nn.Sequential(
            # nn.Linear(self.wsi_projection_dim , int(self.wsi_projection_dim // 4)),
            nn.Linear(256+64, 64),
            # nn.Linear(16, 64),
            nn.ReLU(),
            # nn.Dropout(dropout), ### 
            nn.Linear(64, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.clinical_fc = nn.Sequential(
            nn.Linear(19, 32),  # Match checkpoint: 19->32
            nn.ELU(),  ### ELU
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64)  # Match checkpoint: 32->64
        )
    def forward(self, **kwargs):
        wsi_features = kwargs['x_wsi'] 

        img_data = kwargs['x_img']

        img_features = self.seq_vit_model(img_data)

        clinical_features = kwargs['clinical_data'] # ##   #### here   
        
        mask = None
        return_attn = bool(kwargs.get('return_attn', False))

        # WSI
        wsi_proj = self.wsi_projection_net(wsi_features)

        # # img
        img_proj = self.img_projection_net(img_features)

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

        # WSI             
        if clinical_features is not None:
            assert isinstance(clinical_features, torch.Tensor), "Clinical features should be a torch.Tensor."

            # Clinical features already have batch dim (1, 19), no need to unsqueeze
            clinical_proj = clinical_features.to(torch.float)
            if len(clinical_proj.shape) == 1:
                clinical_proj = clinical_proj.unsqueeze(0)
            clinical_proj = self.clinical_fc(clinical_proj)
            clinical_embedding = self.alpha * clinical_proj
            embedding = torch.cat([wsi_postSA_embed, paths_postSA_embed,clinical_embedding], dim=1)
        else:
            embedding = torch.cat([wsi_postSA_embed, paths_postSA_embed], dim=1)
            
        # clinical_proj = clinical_features.to(torch.float).unsqueeze(0)
        # clinical_proj = self.clinical_fc(clinical_proj)
        # embedding = clinical_proj

        # embedding = torch.cat([wsi_postSA_embed, paths_postSA_embed], dim=1)
        # print("Clinical Features shape after passing to model:", embedding.shape)
        #---> get logits
        logits = self.to_logits(embedding)

        if return_attn:
            return logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return logits
