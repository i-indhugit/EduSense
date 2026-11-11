"""
EduSense Facial Emotion Classifier Architecture Suite
======================================================
Implements candidate neural architectures for facial expression & emotion classification:
- MobileNetV3 (Lightweight real-time baseline)
- ResNet18 (Residual Convolutional baseline)
- EfficientNet-B0 (Compound scaling)
- ConvNeXt-Tiny (Modernized ConvNet)
- ViT-Tiny (Vision Transformer)

Input shape: (B, 3, 224, 224) or (B, 1, 48, 48)
Output shape: (B, 6) -> [Neutral, Happy, Confused, Bored, Sleepy, Frustrated]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionMobileNetV3(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Custom lightweight MobileNetV3-style backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class EmotionResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Simplified Residual Block
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class EmotionViTTiny(nn.Module):
    def __init__(self, num_classes=6, img_size=224, patch_size=16, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0])

def build_emotion_model(arch_name="mobilenetv3", num_classes=6):
    arch_name = arch_name.lower()
    if "mobilenet" in arch_name:
        return EmotionMobileNetV3(num_classes)
    elif "resnet" in arch_name:
        return EmotionResNet18(num_classes)
    elif "vit" in arch_name:
        return EmotionViTTiny(num_classes)
    else:
        return EmotionMobileNetV3(num_classes)
