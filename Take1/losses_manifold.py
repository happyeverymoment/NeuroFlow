import torch
import torch.nn as nn
import torch.nn.functional as F

class SigLIPLoss(nn.Module):
    """
    CLIP-style contrastive loss with two fixes:
      - L2 normalize features before dot product
      - use logit_scale.exp() (common CLIP paramization)
    Drop-in replacement for lavis.models.clip_models.loss.ClipLoss (single-GPU/simple).
    """
    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False):
        super().__init__()
        # keep signature similar to lavis ClipLoss for easy swap
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        """
        image_features, text_features: (N, D)
        logit_scale: nn.Parameter (usually log-scale), will use exp()
        """
        device = image_features.device

        # L2 normalize features (important)
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # common CLIP uses scale = exp(logit_scale)
        try:
            scale = logit_scale.exp()
        except Exception:
            # if logit_scale is float tensor without .exp()
            scale = torch.exp(logit_scale)

        # single-gpu simple case (no distributed gather implemented here)
        logits_per_image = scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        num_logits = logits_per_image.shape[0]
        # prepare labels (0..N-1)
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return loss