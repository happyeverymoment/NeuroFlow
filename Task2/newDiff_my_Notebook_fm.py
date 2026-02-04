
#训练过程
import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

from matplotlib.font_manager import FontProperties
import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
import importlib
try:
    import huggingface_hub as _hf
    # 如果缺少旧 API，就用现有的 hf_hub_download 作为替代
    if not hasattr(_hf, "cached_download") and hasattr(_hf, "hf_hub_download"):
        _hf.cached_download = _hf.hf_hub_download
    if not hasattr(_hf, "cached_path") and hasattr(_hf, "hf_hub_download"):
        # 有些库可能调用 cached_path
        _hf.cached_path = _hf.hf_hub_download
except Exception:
    pass

config = {
"data_path": "/THINGS-Data/EEG/Preprocessed_data_250Hz",
"project": "train_pos_img_text_rep",
"entity": "sustech_rethinkingbci",
"name": "lr=3e-4_img_pos_pro_eeg",
"lr": 3e-4,
"epochs": 50,
"batch_size": 1024,
"logger": True,
"encoder_type":'ATMS',
}
# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 加载图像特征
emb_img_test = torch.load('/EEG_Image_decode/Generation/ViT-H-14_features_test.pt')#只包含图片的文本和image特征 不包含eeg特征
emb_img_train = torch.load('/EEG_Image_decode/Generation/ViT-H-14_features_train.pt')
print("emb_img_test", emb_img_test['img_features'].shape)
print("emb_img_train", emb_img_train['img_features'].shape)
# emb_img_test torch.Size([200, 1024])
# emb_img_train torch.Size([16540, 1024])

import sys
from diffusion_prior_copy import *
from custom_pipeline import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "5" 


emb_img_train_4 = emb_img_train['img_features'].view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)
# 对图像嵌入做 L2 单位化，和 adapter 输出保持同一尺度
emb_img_train_4 = F.normalize(emb_img_train_4, dim=1)

# emb_eeg      = torch.load('/EEG_Image_decode/Generation/ATM_S_eeg_features_sub-08.pt')
# emb_eeg_test = torch.load('/EEG_Image_decode/Generation/ATM_S_eeg_features_sub-08_test.pt')
emb_eeg      = torch.load('/EEG_Image_decode/Generation/NotebookGCN_S_eeg_features_sub-08.pt')
emb_eeg_test = torch.load('/EEG_Image_decode/Generation/NotebookGCN_S_eeg_features_sub-08_test.pt')



# === 新增：简单 linear adapter + 快速短训 (few epochs) ===
EEG_DIM = emb_eeg.shape[1]
CLIP_DIM = emb_img_train_4.shape[1]
adapter = nn.Linear(EEG_DIM, CLIP_DIM).to(device)
opt_ad = torch.optim.Adam(adapter.parameters(), lr=1e-3, weight_decay=1e-5)
# 使用少量数据快速短训，把 EEG 映射到图像 embedding 空间
adapter_epochs = 10
bs = 512
len_train = min(len(emb_eeg), len(emb_img_train_4))
img_targets = emb_img_train_4[:len_train].to(device)
eeg_src = emb_eeg[:len_train].to(device)
for ep in range(adapter_epochs):
    perm = torch.randperm(len_train, device=device)
    for i in range(0, len_train, bs):
        idx = perm[i:i+bs]
        x = eeg_src[idx]
        y = img_targets[idx]
        pred = adapter(x)
        loss_ad = F.mse_loss(pred, y) + 0.1*(1.0 - F.cosine_similarity(pred, y, dim=1).mean())
        opt_ad.zero_grad()
        loss_ad.backward()
        opt_ad.step()

# 将映射结果用于训练 prior（detach 为常量），并进行 L2 单位化
mapped_eeg_train = F.normalize(adapter(emb_eeg.to(device)), dim=1).detach().cpu()
mapped_eeg_test  = F.normalize(adapter(emb_eeg_test.to(device)), dim=1).detach().cpu()

eeg_features_train = mapped_eeg_train

# 创建嵌入数据集对象，用于训练扩散模型
# c_embeddings: 条件嵌入（EEG特征），h_embeddings: 目标嵌入（图像特征）
dataset = EmbeddingDataset(
    c_embeddings=eeg_features_train, h_embeddings=emb_img_train_4, 
    # h_embeds_uncond=h_embeds_imgnet
)
# 创建数据加载器，用于批量处理数据# batch_size=1024: 批次大小为1024# shuffle=True: 每个epoch随机打乱数据# num_workers=64: 使用64个子进程加载数据
# dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)

########################################################################
#新增数据拆分
from torch.utils.data import random_split, DataLoader

# 假设 dataset 已经创建
total_len = len(dataset)
val_len = int(0.01 * total_len)
train_len = total_len - val_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

# dl_train = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=64)
# dl_val = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=64)

dl_train = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
dl_val = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=2, pin_memory=True)

########################################################################

# 创建扩散先验UNet模型
# cond_dim=1024: 条件维度为1024
# dropout=0.1: dropout率为0.1
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.5)
# number of parameters

# 打印模型中需要梯度计算的参数总数（即可训练参数数量）
print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
# 创建管道对象，封装扩散先验模型和设备信息
pipe = Pipe(diffusion_prior, device=device)
pipe.val_dataloader = dl_val  # 设置验证数据加载器
# load pretrained model# 设置模型名称
model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'

#训练25分钟左右 
# 训练模型
# num_epochs=150: 训练150个epoch
# learning_rate=1e-3: 学习率为0.001
#训练部分注释掉了
pipe.train(dl_train, num_epochs=10, learning_rate=1e-3) # Flow Matching训练，已自动适配
# 加载预训练模型权重（被注释掉，如需使用可取消注释）
# pipe.diffusion_prior.load_state_dict(torch.load("/EEG_Image_decode/Generation/fintune_ckpts/ATMS/sub-08/diffusion_prior.pt", map_location=device))

# "/EEG_Image_decode/Generation/fintune_ckpts/ATMS/sub-08/diffusion_prior.pt"
# pipe.diffusion_prior.load_state_dict(torch.load(f'./fintune_ckpts/{config['encoder_type']}/{sub}/{model_name}.pt', map_location=device))

save_path = f'/EEG_Image_decode/Generation/fintune_ckpts/{config["encoder_type"]}/NotebookGCNsub-08FM_test10_epoch25/{model_name}.pt'
# 构建模型保存路径
# 获取路径中的目录部分
directory = os.path.dirname(save_path)
# Create the directory if it doesn't exist
# 如果目录不存在则创建目录，exist_ok=True表示如果目录已存在不会抛出异常
os.makedirs(directory, exist_ok=True)
# 保存扩散先验模型的状态字典到指定路径
torch.save(pipe.diffusion_prior.state_dict(), save_path)
# 保存 adapter 权重以便推理时加载
adapter_path = os.path.join(directory, 'adapter.pt')
torch.save(adapter.state_dict(), adapter_path)


