import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
 
# 从工程中导入模型与数据集（与训练脚本保持一致）
# from ATME_retrieval_newloss_GCN import ATM_E 
from ATME_retrieval_newloss_GCN_other_firstGNN import ATM_E
from eegdatasets_leaveone import EEGDataset

def compute_batch_adjs(model, eeg_batch, device):
    # 输入 eeg_batch: [B, C, T]
    with torch.no_grad():
        eeg_batch = eeg_batch.to(device)
        # x = model.attention_model(eeg_batch)    # [B, C, T]
        x = model.dynamic_gnn(eeg_batch)                # [B, C, T]
        # x=eeg_batch

        x_mean = x.mean(dim=2)                  # [B, C]
        # outer product per sample -> [B, C, C]
        adj = torch.matmul(x_mean.unsqueeze(2), x_mean.unsqueeze(1))
        adj = torch.softmax(adj, dim=-1)
        return adj.cpu().numpy()  # numpy [B, C, C]

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # 加载模型
    model = ATM_E(63, 250)  # 与训练时一致的 init
    model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # 尝试剥离可能的 module. 前缀
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '') if k.startswith('module.') else k
            new_state[nk] = v
        model.load_state_dict(new_state, strict=False)

    model.eval()

    # 加载数据集（使用 test 集合以查看训练后行为）
    dataset = EEGDataset(args.data_path, subjects=[args.subject], train=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    C = 63
    accumulated_adj = np.zeros((C, C), dtype=np.float64)
    total_samples = 0
    sample_count = 0

    # 可选：保存若干单样本邻接图
    per_sample_dir = os.path.join(args.outdir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    for batch_idx, (eeg_data, labels, *rest) in enumerate(loader):
        if sample_count >= args.num_samples:
            break
        # eeg_data shape [B, C, T]
        adj_batch = compute_batch_adjs(model, eeg_data, device)  # [B, C, C]
        B = adj_batch.shape[0]
        # 累加并记录
        accumulated_adj += adj_batch.sum(axis=0)
        total_samples += B

        # 保存前 few_single 保存每个样本邻接图（可视化检查）
        if batch_idx < args.save_per_batches:
            for i in range(min(B, args.save_per_sample_per_batch)):
                idx_global = sample_count + i
                if idx_global >= args.num_samples:
                    break
                fig, ax = plt.subplots(figsize=(6,6))
                sns.heatmap(adj_batch[i], vmin=0, vmax=adj_batch.max(), cmap='viridis', ax=ax)
                ax.set_title(f"Adj sample {idx_global} (batch {batch_idx})")
                fig.savefig(os.path.join(per_sample_dir, f"adj_sample_{idx_global}.png"), bbox_inches='tight')
                plt.close(fig)

        sample_count += B

    if total_samples == 0:
        raise RuntimeError("No samples processed. 检查 dataset/loader 设置。")

    mean_adj = accumulated_adj / total_samples  # [C, C]

    # 对称化（可选）：mean_adj_sym = (mean_adj + mean_adj.T) / 2
    mean_adj_sym = (mean_adj + mean_adj.T) / 2.0

    # 通道重要性：度/强度（行和）
    channel_importance = mean_adj_sym.sum(axis=1)  # [C,]

    # -- 绘图并保存 --
    # 1) 平均邻接矩阵热力图
    plt.figure(figsize=(8,8))
    sns.heatmap(mean_adj_sym, cmap='viridis')
    plt.title("Average adjacency (symmetrized)")
    plt.xlabel("channel")
    plt.ylabel("channel")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "average_adj.png"), dpi=200)
    plt.close()

    # 2) 通道重要性条形图
    plt.figure(figsize=(10,4))
    channels = np.arange(1, C+1)
    importance_norm = channel_importance / (channel_importance.max() + 1e-12)
    plt.bar(channels, importance_norm)
    plt.xlabel("channel")
    plt.ylabel("normalized importance")
    plt.title("Channel importance (degree / normalized)")
    plt.ylim(0.7, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "channel_importance.png"), dpi=200)
    plt.close()

    # 3) 保存数值结果为 npz
    np.savez(os.path.join(args.outdir, "adj_and_importance.npz"),
             mean_adj=mean_adj, mean_adj_sym=mean_adj_sym, channel_importance=channel_importance,
             total_samples=total_samples)

    print(f"Saved average adj and channel importance to {args.outdir} (samples used: {total_samples})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,default= "/Retrieval/models/ATME_retrieval_newloss_GCN_other_firstGNN/sub-01_90.pth",  help="训练好的模型文件路径（.pth/.pt）")
    parser.add_argument("--data_path", type=str, default="/Retrieval/THINGS-Data/EEG/Preprocessed_data_250Hz",help="预处理数据路径")
    parser.add_argument("--subject", type=str, default="sub-01", help="受试者 id")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1024, help="用于统计平均邻接的样本数")
    parser.add_argument("--save_per_batches", type=int, default=2, help="保存前若干 batch 的单样本邻接图")
    parser.add_argument("--save_per_sample_per_batch", type=int, default=4, help="每个保存的 batch 中保存多少样本的邻接图")
    parser.add_argument("--outdir", type=str, default="/Retrieval/adj_vis/firstGNN3", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:6", help="设备")
    args = parser.parse_args()
    main(args)