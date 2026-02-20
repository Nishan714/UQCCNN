import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class ClassicalBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attn1 = ChannelAttention(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn2 = ChannelAttention(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attn3 = ChannelAttention(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.attn4 = ChannelAttention(256)

        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.attn5 = ChannelAttention(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(F.relu(self.attn1(self.bn1(self.conv1(x)))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.attn2(self.bn2(self.conv2(x)))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.attn3(self.bn3(self.conv3(x)))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.attn4(self.bn4(self.conv4(x)))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.attn5(self.bn5(self.conv5(x)))))
        x = self.dropout(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        classical_features = F.relu(self.bn_fc1(self.fc1(x)))
        return classical_features