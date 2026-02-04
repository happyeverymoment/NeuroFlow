import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
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
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch.utils.data import Dataset


class DiffusionPrior(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=1024,
            layers_per_block=4, 
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            time_embed_dim,
            hidden_dim,
        )

        # 2. conditional embedding 
        self.cond_embedding = nn.Linear(cond_dim, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn(),
        )

        # 3.2 hidden
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout),
                )
                for _ in range(layers_per_block)
            ]
        )

        # 3.3 output
        self.output_layer = nn.Linear(hidden_dim, embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)
        t = self.time_embedding(t) # (batch_size, hidden_dim)

        # 2. conditional embedding 
        c = self.cond_embedding(c) if c is not None else 0 # (batch_size, hidden_dim)

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 

        # 3.2 hidden
        for layer in self.hidden_layers:
            x = x + t + c
            x = layer(x) + x
            
        # 3.3 output
        x = self.output_layer(x)

        return x


class DiffusionPriorUNet(nn.Module):

    def __init__(
            self, 
            embed_dim=1024, 
            cond_dim=42,
            hidden_dim=[1024, 512, 256, 128, 64],
            time_embed_dim=512,
            act_fn=nn.SiLU,
            dropout=0.0,
        ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim

        # 1. time embedding
        self.time_proj = Timesteps(time_embed_dim, True, 0)

        # 2. conditional embedding 
        # to 3.2, 3,3

        # 3. prior mlp

        # 3.1 input
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim[0]),
            nn.LayerNorm(hidden_dim[0]),
            act_fn(),
        )

        # 3.2 hidden encoder
        self.num_layers = len(hidden_dim)
        self.encode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1)]
        ) # d_0, ..., d_{n-1}
        self.encode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1)]
        )
        self.encode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i+1]),
                    nn.LayerNorm(hidden_dim[i+1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1)]
        )

        # 3.3 hidden decoder
        self.decode_time_embedding = nn.ModuleList(
            [TimestepEmbedding(
                time_embed_dim,
                hidden_dim[i],
            ) for i in range(self.num_layers-1,0,-1)]
        ) # d_{n}, ..., d_1
        self.decode_cond_embedding = nn.ModuleList(
            [nn.Linear(cond_dim, hidden_dim[i]) for i in range(self.num_layers-1,0,-1)]
        )
        self.decode_layers = nn.ModuleList(
            [nn.Sequential(
                    nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                    nn.LayerNorm(hidden_dim[i-1]),
                    act_fn(),
                    nn.Dropout(dropout),
                ) for i in range(self.num_layers-1,0,-1)]
        )

        # 3.4 output
        self.output_layer = nn.Linear(hidden_dim[0], embed_dim)
        

    def forward(self, x, t, c=None):
        # x (batch_size, embed_dim)
        # t (batch_size, )
        # c (batch_size, cond_dim)

        # 1. time embedding
        t = self.time_proj(t) # (batch_size, time_embed_dim)

        # 2. conditional embedding 
        # to 3.2, 3.3

        # 3. prior mlp

        # 3.1 input
        x = self.input_layer(x) 

        # 3.2 hidden encoder
        hidden_activations = []
        for i in range(self.num_layers-1):
            hidden_activations.append(x)
            t_emb = self.encode_time_embedding[i](t) 
            c_emb = self.encode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.encode_layers[i](x)
        
        # 3.3 hidden decoder
        for i in range(self.num_layers-1):
            t_emb = self.decode_time_embedding[i](t)
            c_emb = self.decode_cond_embedding[i](c) if c is not None else 0
            x = x + t_emb + c_emb
            x = self.decode_layers[i](x)
            x += hidden_activations[-1-i]
            
        # 3.4 output
        x = self.output_layer(x)

        return x


class EmbeddingDataset(Dataset):

    def __init__(self, c_embeddings, h_embeddings):
        self.c_embeddings = c_embeddings
        self.h_embeddings = h_embeddings

    def __len__(self):
        return len(self.c_embeddings)

    def __getitem__(self, idx):
        return {
            "c_embedding": self.c_embeddings[idx],
            "h_embedding": self.h_embeddings[idx]
        }

class EmbeddingDatasetVICE(Dataset):
    def __init__(self, path_data):
        image_features_dict = torch.load(os.path.join(path_data, 'openclip_emb/image_features.pt'))
        self.embedding_vise = torch.load(os.path.join(path_data, 'variables/embedding_vise.pt'))
        self.image_features = image_features_dict['image_features']
        self.labels = image_features_dict['labels']
        self.label2index = image_features_dict['l2i']

    def __len__(self):
        return len(self.image_features)

    def __getitem__(self, idx):
        idx_c = self.label2index[self.labels[idx]]
        return {
            "c_embedding": self.embedding_vise[idx_c],
            "h_embedding": self.image_features[idx]
        }
    

# Copied from diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.add_noise
def add_noise_with_sigma(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.FloatTensor,
) -> torch.FloatTensor:
    # Make sure sigmas and timesteps have the same device and dtype as original_samples
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
        # mps does not support float64
        schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
        timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    else:
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

    step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
        sigma = sigma.unsqueeze(-1)

    noisy_samples = original_samples + noise * sigma
    return noisy_samples, sigma


# diffusion pipe
class Pipe:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
            # self.scheduler.add_noise_with_sigma = add_noise_with_sigma.__get__(self.scheduler)
        else:
            self.scheduler = scheduler
            
        self.device = device
        self.val_dataloader = None
        
    # def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
    #     self.diffusion_prior.train()
    #     device = self.device
    #     optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
    #     for epoch in range(num_epochs):
    #         loss_sum = 0
    #         for batch in dataloader:
    #             c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
    #             h_embeds = batch['h_embedding'].to(device)
    #             N = h_embeds.shape[0]
    #             # Flow Matching: t~U[0,1]，修正为1d
    #             t = torch.rand(N, device=device)  # shape [N]
    #             x_t = c_embeds + t.unsqueeze(1) * (h_embeds - c_embeds)
    #             v_target = h_embeds - c_embeds
    #             v_pred = self.diffusion_prior(x_t, t, c_embeds)
    #             loss = F.mse_loss(v_pred, v_target)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
    #             optimizer.step()
    #             loss_sum += loss.item()
    #         loss_epoch = loss_sum / len(dataloader)

    #         # loss_epoch = loss_sum / len(dataloader)
    #         print(f'epoch: {epoch}, loss: {loss_epoch}')
    #         if hasattr(self, 'val_dataloader') and self.val_dataloader is not None:
    #             self.validate(self.val_dataloader)

    def train(self, dataloader, num_epochs=10, learning_rate=1e-4, use_amp=True, use_flow_matching=True, micro_batch_steps=1):
        self.diffusion_prior.train()
        device = self.device
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        # scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith('cuda'))
        # 兼容 device 为 str 或 torch.device
        device_is_cuda = (isinstance(device, str) and device.startswith('cuda')) or (isinstance(device, torch.device) and device.type == 'cuda')
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_is_cuda)
       
        
        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]
                optimizer.zero_grad()
                # 
                for mb_start in range(0, N, max(1, N//micro_batch_steps)):
                    mb_end = min(N, mb_start + max(1, N//micro_batch_steps))
                    c_mb = c_embeds[mb_start:mb_end] if c_embeds is not None else None
                    h_mb = h_embeds[mb_start:mb_end]
                    # 
                    t = torch.rand(h_mb.shape[0], device=device)
                    x_t = c_mb + t.unsqueeze(1) * (h_mb - c_mb) if c_mb is not None else t.unsqueeze(1) * h_mb
                    v_target = h_mb - c_mb if c_mb is not None else h_mb
                    # with torch.cuda.amp.autocast(enabled=use_amp and device.startswith('cuda')):
                    with torch.cuda.amp.autocast(enabled=use_amp and device_is_cuda):
                        v_pred = self.diffusion_prior(x_t, t, c_mb)
                        if use_flow_matching:
                            loss = F.mse_loss(v_pred, v_target)
                        else:
                            # 
                            loss = F.mse_loss(v_pred + x_t, h_mb) + (1.0 - F.cosine_similarity(v_pred, v_target, dim=1).mean())
                    scaler.scale(loss).backward()
                # 
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item()
            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch:.6f}')
            if hasattr(self, 'val_dataloader') and self.val_dataloader is not None:
                self.validate(self.val_dataloader)

    def validate(self, dataloader):
        self.diffusion_prior.eval()
        device = self.device
        loss_sum = 0
        cos_sum = 0
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                c_embeds = batch['c_embedding'].to(device) if 'c_embedding' in batch.keys() else None
                h_embeds = batch['h_embedding'].to(device)
                N = h_embeds.shape[0]
                t = torch.rand(N, device=device)
                x_t = c_embeds + t.unsqueeze(1) * (h_embeds - c_embeds)
                v_target = h_embeds - c_embeds
                v_pred = self.diffusion_prior(x_t, t, c_embeds)
                loss = F.mse_loss(v_pred, v_target, reduction='sum')
                cos_sim = F.cosine_similarity(v_pred, v_target, dim=1).sum().item()
                loss_sum += loss.item()
                cos_sum += cos_sim
                count += N
        avg_loss = loss_sum / count
        avg_cos = cos_sum / count
        print(f'Validation MSE: {avg_loss:.6f}, Cosine Similarity: {avg_cos:.6f}')
        return avg_loss, avg_cos
    def generate(self, c_embeds=None, num_inference_steps=10, use_amp=False, fallback_cosine_threshold=0.80):

        self.diffusion_prior.eval()
        device = self.device
        device_is_cuda = (isinstance(device, str) and device.startswith('cuda')) or (isinstance(device, torch.device) and getattr(device, 'type', '') == 'cuda')
        with torch.no_grad():
            if c_embeds is None:
                # 
                N = 1
                embed_dim = getattr(self.diffusion_prior, 'cond_dim', None) or getattr(self.diffusion_prior, 'embed_dim', None)
                if embed_dim is None:
                    raise RuntimeError("模型缺少 cond_dim/embed_dim 属性，需提供 c_embeds")
                x = torch.randn(N, embed_dim, device=device, dtype=torch.float32)
                c = x.clone()
            else:
                c = c_embeds.to(device)
                N = c.shape[0]
                x = c.clone()

            for i in range(num_inference_steps):
                t = torch.full((N,), float(i) / max(1, num_inference_steps), device=device, dtype=torch.float32)
                with torch.cuda.amp.autocast(enabled=use_amp and device_is_cuda):
                    v = self.diffusion_prior(x, t, c)
                x = x + v / float(max(1, num_inference_steps))

            try:
                cos = F.cosine_similarity(x, c, dim=1)
                low_mask = cos < float(fallback_cosine_threshold)
                if low_mask.any():
                    x[low_mask] = c[low_mask]
            except Exception:
                pass

        return x

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 1. test prior
    prior = DiffusionPriorUNet(cond_dim=1024)
    x = torch.randn(2, 1024)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 1024)
    y = prior(x, t, c)
    print(y.shape)



