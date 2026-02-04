
#推理过程 
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
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
train = False
classes = None
pictures= None

def load_data():
    data_list = []
    label_list = []
    texts = []
    images = []    
    if train:
        #text_directory = "/THINGS/images_set/training_images"  
        text_directory ="/THINGS-Data/THINGS-EEG_images_set/training_images"
    else:
        #text_directory = "/THINGS/images_set/test_images"
        text_directory ="/THINGS-Data/THINGS-EEG_images_set/test_images"
    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()
    
    if classes is not None:
        dirnames = [dirnames[i] for i in classes]
    for dir in dirnames:
        try:
            idx = dir.index('_')
            description = dir[idx+1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue            
        new_description = f"{description}"
        texts.append(new_description)
    if train:
        # img_directory = "/THINGS/images_set/training_images"
        img_directory ="/THINGS-Data/THINGS-EEG_images_set/training_images"
        
    else:
        # img_directory ="/THINGS/images_set/test_images"
        img_directory ="/THINGS-Data/THINGS-EEG_images_set/test_images"

    
    all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
    all_folders.sort()

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(all_folders):
                folder = all_folders[class_idx]
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:
        print("Error")
    return texts, images
texts, images = load_data()
# images
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
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

emb_img_train_4 = emb_img_train['img_features'].view(1654,10,1,1024).repeat(1,1,4,1).view(-1,1024)


emb_eeg      = torch.load('/EEG_Image_decode/Generation/NotebookGCN_S_eeg_features_sub-08.pt')
emb_eeg_test = torch.load('/EEG_Image_decode/Generation/NotebookGCN_S_eeg_features_sub-08_test.pt')


eeg_features_train=emb_eeg

# 创建嵌入数据集对象，用于训练扩散模型
# c_embeddings: 条件嵌入（EEG特征），h_embeddings: 目标嵌入（图像特征）
dataset = EmbeddingDataset(
    c_embeddings=eeg_features_train, h_embeddings=emb_img_train_4, 
    # h_embeds_uncond=h_embeds_imgnet
)
# 创建数据加载器，用于批量处理数据# batch_size=1024: 批次大小为1024# shuffle=True: 每个epoch随机打乱数据# num_workers=64: 使用64个子进程加载数据
dl = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=64)

# 创建扩散先验UNet模型
# cond_dim=1024: 条件维度为1024
# dropout=0.1: dropout率为0.1
diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
# number of parameters

# 打印模型中需要梯度计算的参数总数（即可训练参数数量）
print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
# 创建管道对象，封装扩散先验模型和设备信息
pipe = Pipe(diffusion_prior, device=device)

# load pretrained model# 设置模型名称
model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'

#训练25分钟左右 
# 训练模型
# num_epochs=150: 训练150个epoch
# learning_rate=1e-3: 学习率为0.001
#训练部分注释掉了
#pipe.train(dl, num_epochs=150, learning_rate=1e-3) # Flow Matching训练，已自动适配
# 加载预训练模型权重（被注释掉，如需使用可取消注释）
# pipe.diffusion_prior.load_state_dict(torch.load("/EEG_Image_decode/Generation/fintune_ckpts/ATMS/sub-08/diffusion_prior.pt", map_location=device))

# "/EEG_Image_decode/Generation/fintune_ckpts/ATMS/sub-08/diffusion_prior.pt"

###############*********************************************************************
pipe.diffusion_prior.load_state_dict(torch.load(f'/EEG_Image_decode/Generation/fintune_ckpts/ATMS/NotebookGCNsub-08FM_test10_epoch25/diffusion_prior.pt', map_location=device))

# === 新增：加载 adapter 并映射测试集 EEG 嵌入 ===
EEG_DIM = emb_eeg_test.shape[1]
CLIP_DIM = emb_img_test['img_features'].shape[1]
adapter = nn.Linear(EEG_DIM, CLIP_DIM).to(device)
###############*********************************************************************
adapter_path = '/EEG_Image_decode/Generation/fintune_ckpts/ATMS/NotebookGCNsub-08FM_test10_epoch25/adapter.pt'
try:
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.eval()
    mapped_eeg_test = adapter(emb_eeg_test.to(device)).detach().cpu()
    print('Loaded adapter and mapped test EEG embeddings.')
except Exception as e:
    print(f'Could not load adapter at {adapter_path}:', e)
    # fallback: use raw EEG embeddings
    mapped_eeg_test = emb_eeg_test



# 
# ...existing code...
import os
# A. 禁用系统代理并强制离线（推荐，避免尝试连 hf）
for k in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY"):
    os.environ.pop(k, None)
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
# 兼容旧/new huggingface_hub API（可选）
try:
    import huggingface_hub as _hf
    if not hasattr(_hf, "cached_download") and hasattr(_hf, "hf_hub_download"):
        _hf.cached_download = _hf.hf_hub_download
    if not hasattr(_hf, "cached_path") and hasattr(_hf, "hf_hub_download"):
        _hf.cached_path = _hf.hf_hub_download
except Exception:
    pass
# ...existing code...

# 在代码开始处添加
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 加载预训练模型权重（被注释掉的代码行）
# pipe.diffusion_prior.load_state_dict(torch.load(f'./fintune_ckpts/{config['data_path']}/{sub}/{model_name}.pt', map_location=device))
# save_path = f'./fintune_ckpts/{config["encoder_type"]}/sub-08test/{model_name}.pt'
# 构建模型保存路径
# 获取路径中的目录部分
# directory = os.path.dirname(save_path)

# Create the directory if it doesn't exist
# 如果目录不存在则创建目录，exist_ok=True表示如果目录已存在不会抛出异常
# os.makedirs(directory, exist_ok=True)

# 保存扩散先验模型的状态字典到指定路径
# torch.save(pipe.diffusion_prior.state_dict(), save_path)
# 从PIL库导入Image模块，用于图像处理
from PIL import Image
# 导入os模块，用于操作系统相关功能
import os

# Assuming generator.generate returns a PIL Image

# 假设generator.generate返回一个PIL图像对象
# 创建生成器实例，设置推理步数为4，并指定设备
generator = Generator4Embeds(num_inference_steps=4, device=device)

# 定义生成图像的保存目录 
###############*********************************************************************
directory = f"/EEG_Image_decode/Generation/generated_imgs/NotebookGCNsub-08FM_test10_epoch10_num_inference_steps50Cos0/"
# 循环200次，处理每个EEG嵌入 生成图像 这里的200对应测试集的200个样本 sdxl没有任何修改直接当成工具进行使用
test_text_emb=emb_img_test['text_features']
import time

start_all = time.perf_counter()
for k in range(200): 
    # 提取第k个EEG嵌入，保持维度使用k:k+1
    eeg_embeds = mapped_eeg_test[k:k+1]
    # 使用管道生成图像，传入条件嵌入、推理步数和指导比例
    h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=5,fallback_cosine_threshold=0.050)
    #偷懒
    image = generator.generate(h.to(dtype=torch.float16))#本来在循环中的 偷懒以此减少运行时间
    # 对每个EEG嵌入生成10个不同的图像
    for j in range(10):
         # 生成图像，将隐藏状态转换为float16精度        
        # Construct the save path for each image
        # 构建每个图像的保存路径
        path = f'{directory}/{texts[k]}/{j}.png'        # Ensure the directory exists

        # 确保图像保存的目录存在，如果不存在则创建
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save the PIL Image
        # 保存PIL图像到指定路径
        image.save(path)
        # 打印图像保存的路径信息
        print(f'Image saved to {path}')

end_all = time.perf_counter()
total_dur = end_all - start_all
print(f"Total time {total_dur:.4f}s, {total_dur/60:.4f}m")


#50步 Total time 50.0815s, 0.8347m

#50步 Total time 18.4559s, 0.3076m
#20步 Total time 7.7252s, 0.1288m
#10步  Total time 3.7856s, 0.0631m
#5步  Total time 1.9999s, 0.0333m
