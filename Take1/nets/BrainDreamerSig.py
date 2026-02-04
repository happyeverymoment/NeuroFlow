import torch
import torch.nn as nn
import numpy as np
from loss import ClipLoss
from losses_manifold import SigLIPLoss
class BrainDreamerEEGEncoder(nn.Module):
    def __init__(self, 
                 num_channels=63, 
                 time_points=250, 
                 embed_dim=1024, 
                 depth=4,         
                 num_heads=16,     
                 patch_size=25,    
                 mlp_ratio=4., 
                 drop_rate=0.1):
        super().__init__()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = time_points // patch_size
        
        self.patch_embed = nn.Conv1d(
            in_channels=num_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=drop_rate, 
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) 
        x = x.transpose(1, 2) 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

class BrainDreamerSigilpEEGEncoder(nn.Module):
    def __init__(self, 
                 num_channels=63, 
                 time_points=250, 
                 embed_dim=1024, 
                 depth=4,         
                 num_heads=16,     
                 patch_size=25,    
                 mlp_ratio=4., 
                 drop_rate=0.1):
        super().__init__()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = SigLIPLoss()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = time_points // patch_size
        
        self.patch_embed = nn.Conv1d(
            in_channels=num_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=drop_rate, 
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) 
        x = x.transpose(1, 2) 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, num_channels=63, time_points=250, output_dim=1024, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True 
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = SigLIPLoss()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, (hidden, cell) = self.lstm(x)
        last_step_feature = output[:, -1, :] 
        return self.fc(last_step_feature)

import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=512, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            GEGLU(embedding_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(drop_proj),
            ResidualAdd(nn.Sequential(
                GEGLU(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            ResidualAdd(nn.Sequential(
                nn.Linear(proj_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class CNNEncoder(nn.Module):
    def __init__(self, num_channels=63, output_dim=1024):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResBlock1D(64, 128, stride=2)
        self.layer2 = ResBlock1D(128, 256, stride=2)
        self.layer3 = ResBlock1D(256, 512, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = SigLIPLoss()
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x) 
        x = x.flatten(1)     
        return self.fc(x)

import torch
import torch.nn as nn

class Brain2ImageEncoder(nn.Module):
    def __init__(self, input_channels=63, time_steps=250, lstm_hidden_size=128, output_dim=1024):
        super(Brain2ImageEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_channels, 
                            hidden_size=lstm_hidden_size, 
                            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, output_dim),
            nn.ReLU()  
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = SigLIPLoss()

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        lstm_out, _ = self.lstm(x)
        last_time_step_feature = lstm_out[:, -1, :] 
        output = self.classifier(last_time_step_feature)
        return output

class SemanticRegionAwareTransformer(nn.Module):
    def __init__(self, in_channels=63, time_steps=250, num_regions=17, embed_dim=1024):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = SigLIPLoss()
        self.in_channels = in_channels
        self.num_regions = num_regions
        self.total_tokens = in_channels + num_regions
        self.time_steps = time_steps
        self.region_tokens = nn.Parameter(torch.randn(1, num_regions, time_steps))
        kernel_sizes = [125, 9, 5]
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.total_tokens, self.total_tokens, k, padding='same', groups=self.total_tokens),
                nn.BatchNorm1d(self.total_tokens),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        self.gating_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.total_tokens * time_steps, len(kernel_sizes)),
            nn.Softmax(dim=-1)
        )
        self.spatial_pe = nn.Parameter(torch.randn(1, self.total_tokens, time_steps))
        encoder_layer = nn.TransformerEncoderLayer(d_model=time_steps, nhead=10, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gat_elec = nn.MultiheadAttention(embed_dim=time_steps, num_heads=10, batch_first=True)
        self.gat_region = nn.MultiheadAttention(embed_dim=time_steps, num_heads=10, batch_first=True)
        self.conv_elec_agg = nn.Conv2d(1, 1, kernel_size=(in_channels, 1))
        self.conv_region_agg = nn.Conv2d(1, 1, kernel_size=(num_regions, 1))
        self.conv_global_agg = nn.Conv2d(1, 1, kernel_size=(2, 1))
        self.projection = nn.Sequential(
            nn.Linear(time_steps, time_steps * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(time_steps * 2, embed_dim)
        )

    def forward(self, x):
        B, C, T = x.shape
        regions = self.region_tokens.expand(B, -1, -1)
        x_hierarchical = torch.cat([x, regions], dim=1)
        gate_weights = self.gating_fc(x_hierarchical)
        branch_outputs = [conv(x_hierarchical) for conv in self.conv_branches]
        x_ms = torch.zeros_like(x_hierarchical)
        for i, branch_out in enumerate(branch_outputs):
            w = gate_weights[:, i].view(B, 1, 1)
            x_ms += w * branch_out
        x_transformer_in = x_ms + self.spatial_pe
        x_region_aware = self.transformer(x_transformer_in)
        z_elec = x_region_aware[:, :self.in_channels, :]
        z_region = x_region_aware[:, self.in_channels:, :]
        z_elec_gat, _ = self.gat_elec(z_elec, z_elec, z_elec)
        z_region_gat, _ = self.gat_region(z_region, z_region, z_region)
        z_elec_img = z_elec_gat.unsqueeze(1) 
        z_region_img = z_region_gat.unsqueeze(1)
        z_elec_agg = self.conv_elec_agg(z_elec_img)
        z_region_agg = self.conv_region_agg(z_region_img)
        z_concat = torch.cat([z_elec_agg, z_region_agg], dim=2)
        z_global = self.conv_global_agg(z_concat)
        z_flat = z_global.view(B, T)
        out = self.projection(z_flat)
        return out

if __name__ == "__main__":
    model = LSTMEncoder()
    out = model(torch.randn(77, 63, 250))
    print(f"Output Shape: {out.shape}")
    model = CNNEncoder()
    out = model(torch.randn(77, 63, 250))
    print(f"Output Shape: {out.shape}")