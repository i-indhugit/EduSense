"""
EduSense-X Reliability-Aware Cross-Attention Multimodal Fusion Module
======================================================================
Research Core: Evidence-Aware Multimodal Temporal Fusion via Reliability-Guided Cross-Attention.

Combines:
- Modality Feature Embeddings (Gaze, Head Pose, Emotion, Posture, Hand, Drowsiness)
- Modality Quality & Confidence Signals (Face Quality, Pose Visibility, Temporal Stability)

Formulation:
  Quality-Scaled Attention Weights:
    R_m = Modality Reliability Score ∈ [0, 1]
    Q = W_q * Modality_Features
    K = W_k * Reliability_Encodings
    V = W_v * Modality_Features
    Attention_Weights = softmax( (Q * K^T) / sqrt(d_k) + log(R_m + 1e-6) )
    Fused_Representation = sum( Attention_Weights * V )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional


class ReliabilityAwareCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, num_modalities: int = 6, num_classes: int = 3):
        super(ReliabilityAwareCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections per modality to unified embedding space
        # Modalities: [0: Face/Emotion, 1: Gaze, 2: HeadPose, 3: Posture, 4: Hand, 5: Drowsiness]
        self.modality_encoders = nn.ModuleList([
            nn.Linear(7, embed_dim),    # Emotion (7 prob outputs)
            nn.Linear(4, embed_dim),    # Gaze (pitch, yaw, iris_dx, iris_dy)
            nn.Linear(3, embed_dim),    # Head Pose (pitch, yaw, roll)
            nn.Linear(4, embed_dim),    # Posture (shoulder_angle, torso_angle, stability, vertical_align)
            nn.Linear(2, embed_dim),    # Hand (raise_prob, gesture_id)
            nn.Linear(4, embed_dim),    # Drowsiness (EAR, MAR, PERCLOS, blink_freq)
        ])

        # Reliability Gate MLP mapping scalar reliability to reliability embedding
        self.reliability_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim)
        )

        # Multi-Head Cross Attention Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Classifiers for Attention & Engagement
        self.attention_classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)  # [High, Moderate, Low]
        )

        self.engagement_classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # [High, Moderate, Low]
        )

    def forward(self, modality_inputs: List[torch.Tensor], reliability_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_inputs: List of 6 tensors [batch_size, modality_dim]
            reliability_scores: Tensor [batch_size, 6] containing reliability ∈ [0, 1] for each modality

        Returns:
            att_logits: Tensor [batch_size, 3] for Attention classification
            eng_logits: Tensor [batch_size, 3] for Engagement classification
            attn_weights: Tensor [batch_size, 6] normalized cross-attention weights per modality
            fused_representation: Tensor [batch_size, embed_dim]
        """
        batch_size = modality_inputs[0].size(0)

        # 1. Encode modalities to unified embedding space -> [batch_size, 6, embed_dim]
        encoded_modalities = []
        for i, proj in enumerate(self.modality_encoders):
            encoded = proj(modality_inputs[i])  # [batch_size, embed_dim]
            encoded_modalities.append(encoded)

        modality_stack = torch.stack(encoded_modalities, dim=1)  # [batch_size, 6, embed_dim]

        # 2. Encode reliability scores -> [batch_size, 6, embed_dim]
        rel_expanded = reliability_scores.unsqueeze(-1)  # [batch_size, 6, 1]
        rel_embeds = self.reliability_encoder(rel_expanded)  # [batch_size, 6, embed_dim]

        # 3. Cross-Attention Query, Key, Value
        Q = self.q_proj(modality_stack)  # [batch_size, 6, embed_dim]
        K = self.k_proj(modality_stack + rel_embeds)  # Incorporate reliability into Key
        V = self.v_proj(modality_stack)  # [batch_size, 6, embed_dim]

        # 4. Compute Reliability-Masked Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)  # [batch_size, 6, 6]
        
        # Add log-reliability bias to prevent unconfident modalities from dominating
        log_rel = torch.log(reliability_scores.clamp(min=1e-6)).unsqueeze(1)  # [batch_size, 1, 6]
        scores = scores + log_rel

        # Softmax over modalities
        attn_matrix = F.softmax(scores, dim=-1)  # [batch_size, 6, 6]
        
        # Average attention weights across query modalities -> [batch_size, 6]
        attn_weights = attn_matrix.mean(dim=1)  # [batch_size, 6]
        
        # Multiply attention weights with values -> [batch_size, 6, embed_dim]
        weighted_v = torch.matmul(attn_matrix, V)  # [batch_size, 6, embed_dim]
        fused = weighted_v.mean(dim=1)  # Pool across modalities -> [batch_size, embed_dim]
        fused = self.out_proj(fused)

        # 5. Classifications
        att_logits = self.attention_classifier(fused)
        eng_logits = self.engagement_classifier(fused)

        return att_logits, eng_logits, attn_weights, fused


def create_numpy_reliability_fusion(modality_features: Dict[str, List[float]],
                                    reliability_scores: Dict[str, float]) -> Dict[str, any]:
    """
    Numpy-backed reliable fallback execution for fast real-time inference without PyTorch overhead.
    Evaluates dynamic modality reliability weights and returns fused scores, attention, engagement,
    uncertainty, and modality contribution breakdown.
    """
    modalities = ["emotion", "gaze", "head_pose", "posture", "hand", "drowsiness"]
    weights = {}
    total_reliability = 0.0

    # 1. Compute raw weights from reliability
    for m in modalities:
        rel = max(0.0, min(1.0, reliability_scores.get(m, 0.0)))
        weights[m] = rel
        total_reliability += rel

    # Normalize weights
    if total_reliability > 1e-5:
        norm_weights = {m: weights[m] / total_reliability for m in modalities}
    else:
        norm_weights = {m: 0.0 for m in modalities}

    # 2. Modality specific evidence contributions
    # Emotion contribution to attention (Positive for neutral/happy, negative for bored/sleepy/frustrated)
    emo_probs = modality_features.get("emotion", [0.0]*7)
    # Mapping [angry, disgust, fear, happy, sad, surprise, neutral]
    emo_score = (emo_probs[3] * 0.9 + emo_probs[6] * 0.8 + emo_probs[5] * 0.7 - emo_probs[0] * 0.4 - emo_probs[4] * 0.5) * 100.0

    # Gaze contribution (pitch, yaw offset from center)
    gaze_feats = modality_features.get("gaze", [0.0, 0.0, 0.0, 0.0])
    gaze_dist = math.sqrt(gaze_feats[0]**2 + gaze_feats[1]**2)
    gaze_score = max(0.0, 100.0 - gaze_dist * 2.5)

    # Head Pose contribution
    hp_feats = modality_features.get("head_pose", [0.0, 0.0, 0.0])
    hp_dist = math.sqrt(hp_feats[0]**2 + hp_feats[1]**2 + hp_feats[2]**2)
    hp_score = max(0.0, 100.0 - hp_dist * 2.0)

    # Posture contribution
    pos_feats = modality_features.get("posture", [0.0, 0.0, 1.0, 1.0])
    pos_score = max(0.0, min(100.0, pos_feats[2] * 70.0 + pos_feats[3] * 30.0))

    # Hand contribution
    hand_feats = modality_features.get("hand", [0.0, 0.0])
    hand_score = 50.0 + (hand_feats[0] * 40.0)

    # Drowsiness penalty
    drowsy_feats = modality_features.get("drowsiness", [0.28, 0.2, 0.0, 1.0])
    ear_val = drowsy_feats[0]
    drowsy_penalty = 50.0 if ear_val < 0.20 else 0.0

    modality_scores = {
        "emotion": max(0.0, min(100.0, emo_score)),
        "gaze": max(0.0, min(100.0, gaze_score)),
        "head_pose": max(0.0, min(100.0, hp_score)),
        "posture": max(0.0, min(100.0, pos_score)),
        "hand": max(0.0, min(100.0, hand_score)),
        "drowsiness": max(0.0, 100.0 - drowsy_penalty)
    }

    # 3. Reliability-Aware Cross-Modal Fusion
    fused_attention = sum(norm_weights[m] * modality_scores[m] for m in modalities)
    fused_engagement = (0.35 * fused_attention +
                        0.25 * modality_scores["posture"] +
                        0.25 * modality_scores["emotion"] +
                        0.15 * modality_scores["hand"])

    return {
        "fused_attention_score": round(float(fused_attention), 2),
        "fused_engagement_score": round(float(fused_engagement), 2),
        "norm_weights": norm_weights,
        "modality_scores": modality_scores,
        "total_reliability": round(float(total_reliability / 6.0), 3)
    }
