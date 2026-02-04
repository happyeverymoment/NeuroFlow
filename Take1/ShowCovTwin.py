import torch
import os
import matplotlib.pyplot as plt

from eegdatasets_leaveone import EEGDataset
from torch.utils.data import DataLoader

def visualize_covariance(model, dataloader, device, num_batches=10, save_dir='./cov_plots'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    batch_count = 0
    with torch.no_grad():
        for i, (eeg_data, _, _, _, _, _) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            eeg_features = model(eeg_data).float()  # shape: [batch, 1024]
            #
            feats = (eeg_features - eeg_features.mean(dim=0)) / (eeg_features.std(dim=0) + 1e-9)
            cov = torch.matmul(feats.T, feats) / feats.shape[0]  # [1024, 1024]
            cov = cov.cpu().numpy()
            # 
            step = cov.shape[0] // 256
            cov_small = cov[::step, ::step][:256, :256]
            plt.figure(figsize=(6, 6))
            im = plt.imshow(cov_small, cmap='viridis', interpolation='nearest')
            #plt.title(f'EEG Embedding Covariance Matrix (Batch {i})')
            # 
            cbar = plt.colorbar(im, fraction=0.048, pad=0.04)
            # 
            plt.xticks([])
            plt.yticks([])
            save_path = os.path.join(save_dir, f'cov_batch_{i}.png')
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"Saved: {save_path}")

            batch_count += 1
            if batch_count >= num_batches:
                break

if __name__ == "__main__":

    
    from contrast_retrieval import ATM_E
    weight_path = '/models/ATME_oldNet_oldLoss/sub-01_5.pth'
    
    
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    # 
    model = ATM_E(num_channels=63, sequence_length=250)
    model.to(device)
    # 
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"Loaded weights from {weight_path}")

    # 
    data_path = "/THINGS-Data/EEG/Preprocessed_data_250Hz"
    dataset = EEGDataset(data_path, subjects=['sub-01'], train=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 
    visualize_covariance(model, dataloader, device, num_batches=10, save_dir='./Showcov_plots_old')