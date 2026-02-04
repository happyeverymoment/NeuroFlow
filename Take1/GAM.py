import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
import os
import open_clip
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 导入你的模型定义和数据集
# 确保 ATME_retrieval_newloss_GCN_other_2.py 中的 ATM_E 类定义是完整的
# from ATME_retrieval_newloss_GCN import ATM_E 
from ATME_retrieval_newloss_GCN_other_firstGNN import ATM_E
from eegdatasets_leaveone import EEGDataset

# === 1. 定义一个用于计算相似度的包装模型 ===
class SimilarityWrapper(nn.Module):
    """
    这个包装器将 CLIP Image Encoder 包装起来。
    它的目标不是分类，而是最大化与给定 EEG 特征向量的相似度。
    """
    def __init__(self, clip_model, target_eeg_embedding):
        super(SimilarityWrapper, self).__init__()
        self.clip_model = clip_model
        # 冻结 CLIP 参数
        for param in self.clip_model.parameters():
            param.requires_grad = True # GradCAM 需要梯度回传，这里设为 True，但不做 optimizer step
        
        self.target_eeg = target_eeg_embedding.detach().clone()
        self.target_eeg = self.target_eeg / self.target_eeg.norm(dim=-1, keepdim=True)

    def forward(self, x):
        # 获取图像特征
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 计算与目标 EEG 向量的余弦相似度
        # shape: [Batch_size, 1]
        similarity = (image_features * self.target_eeg).sum(dim=-1, keepdim=True)
        return similarity

# === 2. ViT 需要特殊的 Reshape Transform ===
def reshape_transform(tensor):
    """
    ViT 输出是 [Batch, Seq_Len, Dim]。
    不管是 OpenCLIP 还是 HuggingFace，通常 Seq_Len = 1 (CLS) + H*W (Patches)。
    我们需要把 Patch 还原成 2D 网格。
    """
    # 过滤掉 CLS token (第一个 token)
    patches = tensor[:, 1:, :]
    
    # 假设输入是 224x224，Patch Size 是 14，那么 Grid 是 16x16
    # 具体取决于模型，ViT-H-14 刚好是 14x14 patch size
    h = 16 
    w = 16
    
    # 重塑为 [Batch, H, W, Dim] -> [Batch, Dim, H, W]
    result = patches.reshape(tensor.size(0), h, w, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # --- A. 加载 CLIP 模型 (Image Encoder) ---
    print("Loading CLIP (ViT-H-14)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k', device=device
    )
    clip_model.eval()

    # --- B. 加载训练好的 EEG Encoder ---
    print("Loading EEG Encoder...")
    # 请根据你的实际配置修改参数
    eeg_model = ATM_E(num_channels=63, sequence_length=250) 
    eeg_model.to(device)
    
    # 加载权重********************************************
    weight_path = "/Retrieval/models/ATME_retrieval_newloss_GCN_other_firstGNN/sub-01_90.pth" # !!! 修改为你训练好的 .pth 文件路径
    if os.path.exists(weight_path):
        eeg_model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: Weight file {weight_path} not found. Using random weights.")

    eeg_model.eval()

    # --- C. 获取一个数据样本 ---
    # 这里我们只取一个样本演示
    data_path = "/Retrieval/THINGS-Data/EEG/Preprocessed_data_250Hz"
    # 使用你的 dataset 类
    dataset = EEGDataset(data_path, subjects=['sub-01'], train=True)
    
    # 选择一个索引 (例如第 0 个)
    arr=[8,408,731,896,898,975,993,992,2 ,14, 95, 286]
    for i in arr: #range(1000):  # 可视化前 100 个样本
        idx = i*8*4
        eeg_data, label, text, _, img_path, _ = dataset[idx]
        
        print(f"Visualizing for Image: {img_path}")
        print(f"Text description: {text}")

        # 1. 处理 EEG
        eeg_input = eeg_data.unsqueeze(0).to(device) # [1, 63, 250]
        
        # 2. 处理图片
        orig_img = Image.open(img_path).convert('RGB')
        orig_img.save(f"/Retrieval/GAM_our/00_orig_{i}.jpg")
        input_tensor = preprocess(orig_img).unsqueeze(0).to(device) # [1, 3, 224, 224]
        
        # 归一化用于显示
        rgb_img = np.float32(orig_img.resize((224, 224))) / 255
        
        # --- D. 计算 EEG Embedding ---
        with torch.no_grad():
            eeg_embedding = eeg_model(eeg_input) # [1, 1024]
            # 如果模型输出不是 1024 维，可能需要投影层，确保与 CLIP 对齐
        
        # --- E. 设置 Grad-CAM ---
        # 包装模型：输入图片，目标是最大化与 eeg_embedding 的相似度
        model_wrapper = SimilarityWrapper(clip_model, eeg_embedding)
        
        # 确定目标层：对于 ViT，通常是最后一个 ResBlock 的 LayerNorm
        target_layers = [clip_model.visual.transformer.resblocks[-1].ln_1]

        cam = GradCAM(model=model_wrapper, target_layers=target_layers, reshape_transform=reshape_transform)

        # --- F. 生成 Heatmap ---
        # targets=None 表示我们不指定类别，因为我们在 forward 里已经计算了相似度
        # 但 GradCAM 库通常预期一个 ClassifierOutputTarget。
        # 由于我们的 forward 输出就是单一数值(相似度)，我们可以通过这类 Target 来传递梯度
        
        # 自定义 target，直接最大化输出的第一个值（即相似度）
        class SimilarityTarget:
            def __call__(self, model_output):
                return model_output
                
        grayscale_cam = cam(input_tensor=input_tensor, targets=[SimilarityTarget()])
        grayscale_cam = grayscale_cam[0, :]

        # 叠加到原图
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # 保存结果
        save_name = f"/Retrieval/GAM_our/11_{i}.jpg"
        cv2.imwrite(save_name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {save_name}")

if __name__ == "__main__":
    main()