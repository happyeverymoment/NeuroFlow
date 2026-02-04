import os
import torch
import torch.optim as optim
import argparse
try:
    Tensor = torch.Tensor
except Exception:
    Tensor = object

import datetime
import math
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset
from einops.layers.torch import Rearrange, Reduce
from lavis.models.clip_models.loss import ClipLoss
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from losses_manifold import SigLIPLoss

# --- SigLIP: a minimal sigmoid-based CLIP-like loss for quick experiments ---
# class SigLIP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce = BCEWithLogitsLoss()

#     def forward(self, eeg_feats, img_feats, logit_scale):
#         # eeg_feats: (N, D), img_feats: (N, D)
#         # compute similarity matrix and apply BCE with identity matrix as positives
#         logits = logit_scale * (eeg_feats @ img_feats.t())
#         targets = torch.eye(logits.size(0), device=logits.device)
#         loss = self.bce(logits, targets)
#         return loss

# --- Barlow Twins style redundancy reduction loss ---

def barlow_twins_loss(z, lambd_offdiag=0.005, eps=1e-9):
    N, D = z.shape
    z_norm = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + eps)
    c = (z_norm.t() @ z_norm) / N
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = c - torch.diag(torch.diag(c))
    off_diag_loss = off_diag.pow(2).sum()
    loss = on_diag + lambd_offdiag * off_diag_loss
    return loss

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
import csv


# 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x

# 
class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)


# 
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shape = (63, 250)
        self.tsconv = EEGNetv4(
            in_chans=self.shape[0],
            n_classes=1440,
            input_window_samples=self.shape[1],
            final_conv_length='auto',
            pool_mode='mean',
            F1=8,
            D=20,
            F2=160,
            kernel_length=4,
            third_kernel_size=(4, 2),
            drop_prob=0.25
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(3)
        x = self.tsconv(x)
        return x

# 
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# 
class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

#
class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

# 
# class Proj_eeg(nn.Sequential):
#     def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
#         super().__init__(
#             nn.Linear(embedding_dim, proj_dim),#输入1440输出1024
#             ResidualAdd(nn.Sequential(
#                 nn.GELU(),#激活函数
#                 nn.Linear(proj_dim, proj_dim),#输入1024输出1024 
#                 nn.Dropout(drop_proj),#随机失活
#             )),
#             nn.LayerNorm(proj_dim),#归一化
#         )
# 
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
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


import torch.nn.functional as F
class DynamicResGNNLayer(nn.Module):
    def __init__(self, channels, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(channels, channels)
        self.alpha = alpha
    def forward(self, x):
        x_mean = x.mean(dim=2)
        adj = torch.matmul(x_mean.unsqueeze(2), x_mean.unsqueeze(1))
        adj = F.softmax(adj, dim=-1)
        x_gcn = torch.matmul(adj, x)
        x_gcn = self.fc(x_gcn.transpose(1,2)).transpose(1,2)
        out = self.alpha * x + (1 - self.alpha) * x_gcn
        return out
# 
import torch
import torch.nn as nn
import torch.nn.functional as F


# class NeuroStageAwareRefinementLayer(nn.Module):
#     """
#     Plug-and-play EEG refinement layer
#     Input / Output shape: [B, C, T]
#     """
#     def __init__(self, channels=63, seq_len=250, alpha=0.8):
#         super().__init__()
#         self.alpha = alpha
#         self.channels = channels
#         self.seq_len = seq_len

#         # ---------- Stage assignment (soft) ----------
#         # From global EEG response -> 3 neuro stages
#         self.stage_proj = nn.Linear(seq_len, 3)

#         # ---------- Early stage: local temporal / frequency bias ----------
#         self.early_conv = nn.Conv1d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=7,
#             padding=3,
#             groups=channels  # depthwise
#         )

#         # ---------- Mid stage: channel interaction (lightweight) ----------
#         self.mid_fc1 = nn.Conv1d(channels, channels // 4, kernel_size=1)
#         self.mid_fc2 = nn.Conv1d(channels // 4, channels, kernel_size=1)

#         # ---------- Late stage: global temporal modeling ----------
#         self.late_attn = nn.MultiheadAttention(
#             embed_dim=seq_len,
#             num_heads=5,
#             batch_first=True
#         )

#         # ---------- Fusion gate ----------
#         self.stage_gate = nn.Parameter(torch.ones(3))


#     def forward(self, x):
#         """
#         x: [B, C, T]
#         """
#         B, C, T = x.shape

#         # ===== Stage assignment =====
#         # Global summary over channels
#         x_global = x.mean(dim=1)              # [B, T]
#         stage_weight = F.softmax(
#             self.stage_proj(x_global), dim=-1
#         )                                      # [B, 3] shape = torch.Size([1024, 3])

#         # Expand weights
#         w1 = stage_weight[:, 0].view(B, 1, 1)# shape = torch.Size([1024, 1, 1])
#         w2 = stage_weight[:, 1].view(B, 1, 1)
#         w3 = stage_weight[:, 2].view(B, 1, 1)

#         # ===== Early stage =====
#         x_early = self.early_conv(x)

#         # ===== Mid stage =====
#         w_mid = F.adaptive_avg_pool1d(x, 1)
#         w_mid = F.relu(self.mid_fc1(w_mid))
#         w_mid = torch.sigmoid(self.mid_fc2(w_mid))
#         x_mid = x * w_mid

#         # ===== Late stage =====
#         # Treat channels as tokens, time as embedding
#         x_late, _ = self.late_attn(x, x, x)

#         # ===== Stage fusion =====
#         x_refined = (
#             self.stage_gate[0] * w1 * x_early +
#             self.stage_gate[1] * w2 * x_mid   +            
#             self.stage_gate[2] * w3 * x_late
#         )

#         # ===== Residual =====
#         out = self.alpha * x + (1 - self.alpha) * x_refined
#         return out

class PrototypeRefinementLayer(nn.Module):
    def __init__(self, embed_dim=1024, num_prototypes=64, temperature=0.07, residual_alpha=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.residual_alpha = residual_alpha
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def init_prototypes(self, visual_features):
        try:
            from sklearn.cluster import KMeans
            if isinstance(visual_features, torch.Tensor):
                features_np = visual_features.detach().cpu().numpy()
            else:
                features_np = visual_features
            kmeans = KMeans(n_clusters=self.num_prototypes, n_init='auto', random_state=42)
            kmeans.fit(features_np)
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
            with torch.no_grad():
                self.prototypes.copy_(centers)
                self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        except ImportError:
            pass

    def forward(self, x):
        q = F.normalize(x, dim=-1)
        k = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(q, k.t()) / self.temperature
        attn_weights = F.softmax(sim, dim=-1)
        x_proto = torch.matmul(attn_weights, self.prototypes)
        out = (1 - self.residual_alpha) * x + self.residual_alpha * x_proto
        out = self.norm(out)
        return out

class NeuroFlow(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(NeuroFlow, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = SigLIPLoss()
        self.dynamic_gnn = DynamicResGNNLayer(num_channels)
        self.proto_refine = PrototypeRefinementLayer(embed_dim=num_latents, num_prototypes=16, temperature=0.07, residual_alpha=0.2)

    def forward(self, x):
        x = self.dynamic_gnn(x)
        x = self.attention_model(x)
        x = self.subject_wise_linear[0](x)
        eeg_embedding = self.enc_eeg(x)
        out = self.proj_eeg(eeg_embedding)
        out = self.proto_refine(out)
        return out
                                                                                                                                            
# 
def train_model(model, dataloader, optimizer, device, text_features_all, img_features_all, bt_weight=0.005):
    model.train()
    text_features_all = text_features_all.to(device).float()
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.99
    features_list = []
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        eeg_features = model(eeg_data).float()
        features_list.append(eeg_features)
        logit_scale = model.logit_scale
        img_loss = model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = model.loss_func(eeg_features, text_features, logit_scale)
        bt_loss = barlow_twins_loss(eeg_features)
        loss = alpha * img_loss + (1 - alpha) * text_loss   + bt_weight * bt_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)
        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    return average_loss, accuracy

def evaluate_model(model, dataloader, device, text_features_all, img_features_all, k):
    model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            eeg_features = model(eeg_data).float()
            logit_scale = model.logit_scale
            img_loss = model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = model.loss_func(eeg_features, text_features, logit_scale)
            loss = img_loss*alpha + text_loss*(1-alpha)
            total_loss += loss.item()
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                logits_single = logits_img
                predicted_label = selected_classes[torch.argmax(logits_single).item()]
                if predicted_label == label.item():
                    correct += 1
                _, top5_indices = torch.topk(logits_single, 5, largest=True)
                if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                    top5_correct_count += 1
                total += 1
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc
# 
def main_train_loop(sub, model, train_dataloader, test_dataloader, optimizer, device,
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    print(f"Using device: {device}")
    if hasattr(model, 'proto_refine'):
        model.proto_refine.init_prototypes(img_features_train_all)
        model.proto_refine.to(device)
    logger = wandb_logger(config) if logger else None
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []
    best_accuracy = 0.0
    best_epoch_info = {}
    results = []
    for epoch in range(config['epochs']):
        train_loss, train_accuracy = train_model(
            model,
            train_dataloader,
            optimizer,
            device,
            text_features_train_all,
            img_features_train_all,
            bt_weight=config.get('bt_weight', 0.005)
        )
        if epoch%10 == 0:
            model_save_dir = "/Retrieval/models/r2g8NeuroFlow_SigTwin_Try2Nsar8r_FirstGNN_notebook_woNSR_noline"
            os.makedirs(model_save_dir, exist_ok=True)
            if config['insubject']==True:
                torch.save(model.state_dict(), f"/Retrieval/models/r2g8NeuroFlow_SigTwin_Try2Nsar8r_FirstGNN_notebook_woNSR_noline/{sub}_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"/Retrieval/models/r2g8NeuroFlow_SigTwin_Try2Nsar8r_FirstGNN_notebook_woNSR_noline/across_{epoch}.pth")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_loss, test_accuracy, top5_acc = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
        _, v2_acc, _ = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 2)
        _, v4_acc, _ = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 4)
        _, v10_acc, _ = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 10)
        _, v50_acc, v50_top5_acc  = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 50)
        _, v100_acc, v100_top5_acc = evaluate_model(model, test_dataloader, device, text_features_test_all, img_features_test_all, k = 100)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        epoch_results = {
        "epoch": epoch + 1,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss
        }
        results.append(epoch_results)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch_results = {
            "epoch": epoch + 1,
            "test_accuracy": test_accuracy,
            "v2_acc": v2_acc,
            "v4_acc": v4_acc,
            "v10_acc": v10_acc,
            "top5_acc":top5_acc,
            "v50_acc": v50_acc,
            "v100_acc": v100_acc,
            "v50_top5_acc":v50_top5_acc,
            "v100_top5_acc": v100_top5_acc,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss
            }
            torch.save(model.state_dict(), f"/Retrieval/models/r2g8NeuroFlow_SigTwin_Try2Nsar8r_FirstGNN_notebook_woNSR_noline/{sub}_{epoch}_best.pth")
        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config['epochs']} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
    print("Best Test Accuracy: {:.4f} at Epoch {}".format(best_accuracy, best_epoch_results['epoch']))
    print("**********************************************")
    print("Best Epoch Results:", best_epoch_results)
    print("**********************************************")
    results.append(best_epoch_results)
    return results

def main():
    parser = argparse.ArgumentParser(description='Train EEG-Image/Text Model')
    parser.add_argument('--data_path', type=str, default="/THINGS-Data/EEG/Preprocessed_data_250Hz", help='Path to the preprocessed data')
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='Project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='Entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--logger', action='store_true', help='Enable logging')
    parser.add_argument('--insubject',default=True, action='store_true', help='Train within subject')
    parser.add_argument('--encoder_type', type=str, default='NeuroFlow', help='EEG encoder model type')
    parser.add_argument('--save_name', type=str, default='r2g8NeuroFlow_SigTwin_Try2Nsar8r_FirstGNN_notebook_woNSR_noline_sub8', help='EEG encoder model type, you can choose from these options: Projector, EEGConformer_Encoder, MetaEEG, EEGNetv4_Encoder, ShallowFBCSPNet_Encoder, NICE, ATCNet_Encoder, EEGITNet_Encoder')
    parser.add_argument('--device', type=str, default='cuda:4', help='Device to use for training (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--bt_weight', type=float, default=0.005, help='Weight for Barlow Twins redundancy reduction loss')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_path = args.data_path
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    for sub in subjects:
        model = globals()[args.encoder_type](63, 250)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print(f'Processing {sub}: number of parameters:', sum(p.numel() for p in model.parameters()))
        train_dataset = EEGDataset(
            data_path,
            subjects=[sub] if args.insubject else [],
            exclude_subject=sub if not args.insubject else None,
            train=True
        )
        test_dataset = EEGDataset(
            data_path,
            subjects=[sub] if args.insubject else [],
            exclude_subject=sub if not args.insubject else None,
            train=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features
        config = vars(args)
        results = main_train_loop(
            sub,
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            text_features_train_all,
            text_features_test_all,
            img_features_train_all,
            img_features_test_all,
            config,
            logger=args.logger
        )
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        results_dir = f"./outputs/{args.save_name}/{sub}/{current_time}"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/{args.save_name}_{'cross_exclude_' if not args.insubject else ''}{sub}.csv"
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()

