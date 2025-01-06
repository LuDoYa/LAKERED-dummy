import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage import img_as_float
from einops import rearrange
import cv2

# First, let's add the necessary components from FKRM

def show_debug(tensor, name="THIS TENSOR"):
    if isinstance(tensor, torch.Tensor):
        print(f"{name} IS A TENSOR OF - {tensor}"
                f"WITH SHAPE: {tensor.shape}, "
                f"MEAN: {tensor.mean()}, "
                f"STD: {tensor.std()}, "
                f"MIN: {tensor.min()}, "
                f"MAX: {tensor.max()}")
    else:
        print(f"{name} IS {tensor}")

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        show_debug(x, "AFTER LINEAR LAYER 1")
        x = self.act(x)
        show_debug(x, "AFTER ACTIVATION LAYER")
        x = self.drop(x)
        show_debug(x, "AFTER DROPOUT LAYER 1")
        x = self.fc2(x)
        show_debug(x, "AFTER LINEAR LAYER 2")
        x = self.drop(x)
        show_debug(x, "AFTER DROPOUT LAYER 2")
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x, context):
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        return attn @ v

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    if(len(size) == 4):
        B, C = size[:2]
        feat_var = feat.view(B, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        return feat_mean, feat_std
    if len(size)==5:
        B, C, N = size[:3]
        feat_var = feat.view(B, C, N, -1).var(dim=3) + eps
        feat_std = feat_var.sqrt().view(B, C, N, 1, 1)
        feat_mean = feat.view(B, C, N, -1).mean(dim=3).view(B, C, N, 1, 1)
        return feat_mean, feat_std

class PSF(nn.Module):
    def __init__(self, in_planes):
        super(PSF, self).__init__()
        self.e = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))

    def mean_var_norm(self, feat):
        size = feat.size()
        mean, std = calc_mean_std(feat)
        norm_feat = (feat - mean.expand(size)) / std.expand(size)
        return norm_feat

    def patch_mv_norm(self, feat, n):
        (b,c,h,w) = feat.size()
        eps = 1e-8
        zeroPad2d = torch.nn.ZeroPad2d(n // 2)
        feat_pad = zeroPad2d(feat)
        feat_fold = F.unfold(feat_pad, (n, n), stride=1).view(b,c,n*n,-1)
        feat_mean = feat_fold.mean(dim=2)
        feat_std = feat_fold.var(dim=2).sqrt() + eps
        feat_norm = ((feat.view(b,c,-1) - feat_mean) / feat_std).view(b,c,h,w)
        return feat_norm

    def patch_adain(self, x, y, n):
        (b, c, h, w) = y.size()
        zeroPad2d = torch.nn.ZeroPad2d(n // 2)
        x_norm = self.patch_mv_norm(x,n)

        y_pad = zeroPad2d(y)
        y_fold = F.unfold(y_pad, (n, n), stride=1).view(b, c, n * n, -1)
        y_mean = y_fold.mean(dim=2).view(b,c,h,w)
        y_std = y_fold.var(dim=2).sqrt().view(b,c,h,w)
        x_adain = x_norm * y_std + y_mean
        return x_adain

    def forward(self, front, back, mask):
        winsize = 7
        EE = self.e(self.patch_adain(front, back, winsize))
        show_debug(EE, "EE")
        #Structural feature
        FF = self.f(self.patch_mv_norm(front, winsize))
        show_debug(FF, "FF")
        GG = self.g(self.patch_mv_norm(back, winsize))
        show_debug(GG, "GG")
        #Appearance feature
        HH = self.h(back)
        show_debug(HH, "HH")
        
        b, _, h, w = GG.size()
        FF = FF.view(b, -1, w * h)
        GG = GG.view(b, -1, w * h)
        
        # Compute similarity
        F_n = (FF * FF).sum(dim=1).sqrt()
        show_debug(F_n, "F AFTER NORMALIZED")
        G_n = (GG * GG).sum(dim=1).sqrt()
        show_debug(G_n, "G AFTER NORMALIZED")
        S = torch.mul(FF, GG).sum(dim=1) / (F_n * G_n)
        show_debug(S, "S")
        
        # Normalize similarity
        S_n = ((S - S.min(dim=1)[0].unsqueeze(1)) / 
               (S.max(dim=1)[0].unsqueeze(1) - S.min(dim=1)[0].unsqueeze(1))).view(b, 1, h, w)
        show_debug(S_n, "S AFTER NORMALIZED")
        
        # Fuse features
        fused_features = torch.mul(S_n, EE) + torch.mul(1 - S_n, HH)
        show_debug(fused_features, "PSF RESULT OR FUSED FEATURES")
        
        # Compute loss
        loss = None
        
        return fused_features, loss

    def compute_loss(self, fused_features, Ff, Fb, A):
        """
        Computes the loss for PSF.
        :param fused_features: Output of the fusion (B, C, H, W)
        :param Ff: Front/foreground features (B, C, H, W)
        :param Fb: Back/background features (B, C, H, W)
        :param A: Similarity matrix S_n (B, 1, H, W)
        :return: Combined loss value
        """
        eps = 1e-8  # For numerical stability
        
        # Structure Consistency Loss
        cos_sim = F.cosine_similarity(fused_features, Fb, dim=1)
        structure_loss = 1 - torch.mean(A.squeeze(1) * cos_sim)

        # Appearance Consistency Loss
        local_mean = torch.mean(Fb, dim=[2, 3], keepdim=True)
        local_var = torch.var(Fb, dim=[2, 3], keepdim=True) + eps
        fused_mean = torch.mean(fused_features, dim=[2, 3], keepdim=True)
        fused_var = torch.var(fused_features, dim=[2, 3], keepdim=True) + eps

        appearance_loss = torch.mean((local_mean - fused_mean) ** 2 + 
                                   (local_var - fused_var) ** 2)

        # Combine losses
        total_loss = structure_loss + appearance_loss

        return total_loss


# Background Knowledge Retrieval Augment
class FKRM(nn.Module):
    def __init__(self,
                 LR_config,
                 embed_dim=3,
                 n_embed=8192,
                 debug_viz=False):
        super().__init__()
        self.LR_config = LR_config
        self.rec_loss = LR_config['rec_loss']
        self.loss_type = LR_config['loss_type']
        fg_embed = torch.randn(embed_dim, n_embed)
        self.register_buffer("bg_embed", fg_embed)
        approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
        self.mlp_in = Mlp(3,
                       6,
                       embed_dim,
                       act_layer=approx_gelu, 
                       drop=0)
        self.crossAttn = CrossAttention(embed_dim, embed_dim)
        self.mlp_out = Mlp(embed_dim,
                       6,
                       3,
                       act_layer=approx_gelu, 
                       drop=0)
        self.n_super_pix = int(LR_config['n_super_pix'])
        self.fuse = nn.Conv2d(6,3,kernel_size=1)
        self.PSF = PSF(in_planes=3)
        self.initialize_weights()
        self.debug_viz = debug_viz
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(3)
        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def visualize_step(self, step_name, images, num_cols=4):
        """Helper function to visualize intermediate results"""
        if not self.debug_viz:
            return
            
        num_images = len(images)
        num_rows = (num_images + num_cols - 1) // num_cols
        
        plt.figure(figsize=(15, 3*num_rows))
        for idx, (title, img) in enumerate(images):
            plt.subplot(num_rows, num_cols, idx + 1)
            
            # Convert tensor to numpy and handle different formats
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
                if img.shape[0] == 3:  # CHW format
                    img = np.transpose(img, (1, 2, 0))
            
            # Normalize if needed
            if img.max() > 1 or img.min() < 0:
                img = (img - img.min()) / (img.max() - img.min())
                
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        
        plt.suptitle(step_name)
        plt.tight_layout()
        plt.show()

    # Localized Masked Pooling 
    def LMP(self, bg, mask, n, s):
        print("\n=== Starting LMP Process ===")
        print(f"Input bg shape: {bg.shape}")
        print(f"Input mask shape: {mask.shape}")
        print(f"Number of segments (n): {n}")
        
        res = []
        for b in range(bg.shape[0]):
            print(f"\nProcessing batch {b}:")
            
            # Convert image for SLIC
            img = rearrange(bg[b].cpu(), 'c h w -> h w c')
            print(f"Rearranged image shape: {img.shape}")
            print(f"Image stats - mean: {img.mean():.6f}, std: {img.std():.6f}")
            
            # Process mask
            m = 1-mask[b].squeeze().cpu()
            print(f"Mask stats - mean: {m.mean():.6f}, std: {m.std():.6f}")
            print(f"Number of masked pixels: {m.sum()}")
            
            # SLIC segmentation
            segments = slic(img_as_float(img), n_segments=n, sigma=5, mask=m)
            print(f"Segments shape: {segments.shape}")
            print(f"Unique segment values: {np.unique(segments)}")
            
            temp_seg = []
            for value in range(1, n+1):
                value_indices = (segments == value)
                if value_indices.any():
                    img_subset = bg[b][:, value_indices]
                    print(f"\nSegment {value}:")
                    print(f"Number of pixels in segment: {value_indices.sum()}")
                    print(f"Subset shape: {img_subset.shape}")
                    
                    avg_pooled = img_subset.mean(dim=1)
                    print(f"Pooled values: {avg_pooled}")
                    temp_seg.append(avg_pooled)
                else:
                    print(f"\nSegment {value} is empty, using zeros")
                    temp_seg.append(torch.tensor([0, 0, 0]).to(bg.device))
            
            batch_segments = torch.stack(temp_seg)
            print(f"\nBatch segments shape: {batch_segments.shape}")
            print(f"Batch segments stats - mean: {batch_segments.mean():.6f}, std: {batch_segments.std():.6f}")
            res.append(batch_segments)
        
        final_result = torch.stack(res)
        print(f"\nFinal LMP output shape: {final_result.shape}")
        print(f"Final output stats - mean: {final_result.mean():.6f}, std: {final_result.std():.6f}")
        print("=== LMP Process Complete ===\n")
        
        return final_result
    
    def forward(self, cond, x_start):
        bg, mask = cond[0].split(3, 1) # b, 3, 128, 128
        
        # Visualize input
        if self.debug_viz:
            self.visualize_step("Input", [
                ("Background", bg[0]),
                ("Mask", mask[0].squeeze()),
                ("x_start", x_start[0])
            ])

        # Extremely small object
        if not self.training:
            if (1 - mask).sum() == 0:
                return cond, 0
            
        # (1) LMP to extract the background features [vec_fg]
        vec_bg = self.LMP(bg, mask, self.n_super_pix, 5) # b n 3
        
        # Visualize LMP results
        if self.debug_viz:
            # Create visualization of superpixels
            img = rearrange(bg[0].cpu(), 'c h w -> h w c')
            m = 1-mask[0].squeeze().cpu()
            segments = slic(img_as_float(img), n_segments=self.n_super_pix, sigma=5, mask=m)
            
            self.visualize_step("LMP Results", [
                ("Superpixel Segmentation", segments)
            ])

        # After LMP
        show_debug(vec_bg, "VEC BG AFTER LMP")

        # (2) BKRM to get the background related features [vec_bg]
        vec_bg_q = self.norm1(self.mlp_in(vec_bg))
        show_debug(vec_bg_q, "VEC BG Q AFTER MLP")
        code_book = self.bg_embed.transpose(1,0).unsqueeze(0).repeat(vec_bg_q.shape[0], 1, 1).to(bg.device)
        fg_emb = self.crossAttn(vec_bg_q, code_book)
        vec_fg = self.norm2(self.mlp_out(fg_emb))
        show_debug(vec_fg, "VEC FG AFTER MLP OUT")

        vec_fg = rearrange(vec_fg, 'b n c -> b c n')
        
        # (3) Reasoning enhancement:
        # upsample->concate->fuse
        vec_fg = vec_fg.unsqueeze(3).expand(-1,-1, -1, self.n_super_pix)

        vec_fg = torch.nn.functional.interpolate(vec_fg, size=[128, 128], mode='nearest')
        
        # Visualize upsampled features
        if self.debug_viz:
            print("vec_fg is ", vec_fg)
            self.visualize_step("Upsampled Features", [
                ("Upsampled Features", vec_fg[0])
            ])

        PSF_output, PSF_loss = self.PSF(vec_fg, bg, mask)
        
        # Visualize PSF results
        if self.debug_viz:
            self.visualize_step("PSF Results", [
                ("PSF Output", PSF_output[0]),
                ("Original Background", bg[0]),
                ("Difference", (PSF_output - bg)[0])
            ])

        # (4) Background feature injection to get new cond
        new_cond = torch.cat((PSF_output, mask), dim=1)
        
        bgrec_loss = None
        if self.rec_loss:
            bgrec_loss = self.get_loss(PSF_output*mask, x_start*mask, mean=False).mean([1, 2, 3])
        
        return [new_cond], bgrec_loss, PSF_loss

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def tensor_stats(self, tensor, name="Tensor"):
        """Helper function to print tensor statistics"""
        return (f"{name} - "
                f"shape: {tensor.shape}, "
                f"mean: {tensor.mean():.6f}, "
                f"std: {tensor.std():.6f}, "
                f"min: {tensor.min():.6f}, "
                f"max: {tensor.max():.6f}")
    
def create_clusterable_image_alternative(size=128, n_regions=16):
    """
    Alternative version with more distinct regions
    """
    # Create base grid
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Create checkerboard-like patterns
    r = np.sin(3 * np.pi * xx) * np.sin(3 * np.pi * yy)
    g = np.cos(4 * np.pi * xx ** 2 + 4 * np.pi * yy ** 2)
    b = np.sin(5 * np.pi * np.sqrt(xx ** 2 + yy ** 2))
    
    # Add some gradients
    r += xx + yy
    g += xx - yy
    b += -xx + yy
    
    # Normalize each channel to [0, 1]
    r = (r - r.min()) / (r.max() - r.min())
    g = (g - g.min()) / (g.max() - g.min())
    b = (b - b.min()) / (b.max() - b.min())
    
    # Stack and convert to tensor
    img = np.stack([r, g, b], axis=0)
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)
    
    return img_tensor

def test_integration():
    # Set random seed
    torch.manual_seed(23)

    create_clusterable_image_alternative(128, 16)
    
    # Create sample inputs
    batch_size = 2
    channels = 3
    height = 128
    width = 128
    
    # Create sample background and mask
    bg = torch.randn(batch_size, channels, height, width)
    mask = torch.rand(batch_size, 1, height, width) > 0.5
    cond = torch.cat([bg, mask.float()], dim=1)
    x_start = torch.randn_like(bg)
    
    # Create config
    LR_config = {
        'rec_loss': True,
        'loss_type': 'l2',
        'n_super_pix': 16,
        'bg_retrieval': True
    }
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cond = cond.to(device)
    x_start = x_start.to(device)
    
    # Initialize FKRM with visualization enabled
    fkrm = FKRM(LR_config, debug_viz=True).to(device)
    
    # Run forward pass
    print("\nStarting test...")
    with torch.no_grad():
        new_cond, bgrec_loss, psf_loss = fkrm([cond], x_start)
    
    print("\nFinal Results:")
    print(f"New condition shape: {new_cond[0].shape}")
    print(f"Background reconstruction loss: {bgrec_loss}")
    print(f"PSF loss: {psf_loss}")

if __name__ == "__main__":
    test_integration()