"""
19CLI - SurvPath
checkpoint
"""

import torch
import torch.nn as nn

class ClinicalOnlyModel(nn.Module):
    def __init__(self, clinical_dim=19, dropout=0.1):
        super().__init__()
        
        # Clinical FC matches trained CLI19 checkpoint: 19 -> 128 -> 64
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)
        )
        
        # Logits: 64 -> 64 -> 1
        self.to_logits = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, **kwargs):
        """
        Forward pass
        kwargs x_wsi, x_img, clinical_data
        clinical_data
        """
        clinical_features = kwargs['clinical_data']
        
        # Clinical pathway
        clinical_embed = self.clinical_fc(clinical_features)
        
        # Get logits
        logits = self.to_logits(clinical_embed)
        
        return logits


class SurvPath_19CLI(ClinicalOnlyModel):
    def __init__(self, dropout=0.1):
        super().__init__(clinical_dim=19, dropout=dropout)
