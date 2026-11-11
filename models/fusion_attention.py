"""
EduSense Intelligence Engine (EIE) - Learnable Multi-Modal Cross-Attention Fusion
===================================================================================
Replaces static manual weighting with a Learnable Multi-Head Cross-Attention Network.

Architecture:
Input: Multi-modal feature matrix F in R^(N x 5 x D) representing:
  - Modality 1: Gaze & Attention vector
  - Modality 2: Facial Emotion Embedding
  - Modality 3: Participation & Gesture vector
  - Modality 4: Posture & Spatial vector
  - Modality 5: Drowsiness & Fatigue indicators

Output:
  - Dynamic Feature Attention Weights alpha = [alpha_att, alpha_emo, alpha_part, alpha_pos, alpha_drowsy]
  - ESI Score prediction in [0, 100]
  - Engagement Percentage in [0, 100]
  - Confusion Probability & Index in [0, 100]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusionEngine(nn.Module):
    def __init__(self, feature_dim=32, num_heads=4, num_modalities=5):
        super().__init__()
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim

        # Projection layers per modality
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(6, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            ) for _ in range(num_modalities)
        ])

        # Multi-Head Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        
        # Self-Attention Feature Importance Weight Generator
        self.importance_net = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, 64),
            nn.ReLU(),
            nn.Linear(64, num_modalities),
            nn.Softmax(dim=-1)
        )

        # Output Prediction Heads
        self.esi_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.engagement_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.confusion_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_features: torch.Tensor):
        """
        input_features shape: (B, 5, 6)
        Returns:
          - esi_score (B, 1) in [0, 100]
          - engagement_pct (B, 1) in [0, 100]
          - confusion_index (B, 1) in [0, 100]
          - feature_importance (B, 5) -> [alpha_gaze, alpha_emo, alpha_part, alpha_pos, alpha_drowsy]
        """
        B = input_features.shape[0]
        projected = []
        for i in range(self.num_modalities):
            proj = self.modality_projections[i](input_features[:, i, :])
            projected.append(proj.unsqueeze(1))

        # Concatenate modal embeddings: (B, 5, D)
        modal_matrix = torch.cat(projected, dim=1)

        # Apply Multi-Head Cross-Attention
        attn_out, _ = self.cross_attention(modal_matrix, modal_matrix, modal_matrix)

        # Compute dynamic feature importance weights alpha
        flattened = attn_out.reshape(B, -1)
        alpha = self.importance_net(flattened) # (B, 5)

        # Weighted pool over modalities
        weighted_fusion = torch.sum(attn_out * alpha.unsqueeze(-1), dim=1) # (B, D)

        # Predict multi-task outputs
        esi_score = self.esi_head(weighted_fusion) * 100.0
        engagement_pct = self.engagement_head(weighted_fusion) * 100.0
        confusion_index = self.confusion_head(weighted_fusion) * 100.0

        return {
            "esi_score": esi_score,
            "engagement_pct": engagement_pct,
            "confusion_index": confusion_index,
            "feature_importance": alpha
        }
