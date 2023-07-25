import math

import torch
import torch.nn as nn
import timm
from timm.models.layers import DropPath, trunc_normal_

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_cfg):
        super().__init__()
        self.encoder_cfg = encoder_cfg
        depth = encoder_cfg.NUM_LAYERS
        dpr = [x.item() for x in torch.linspace(0, encoder_cfg.DROP_PATH_RATE, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=encoder_cfg.NUM_FEATURES, num_heads=encoder_cfg.NUM_HEADS, mlp_hidden_dim=encoder_cfg.NUM_HIDDEN_FEATURES, 
                qkv_bias=False, qk_scale=False, drop=0., attn_drop=0., drop_path = dpr[i]
                )
            for i in range(depth)])

    def forward(self, x, pos):
        x = x + pos
        for _, block in enumerate(self.blocks):
            x = block(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_cfg):
        super().__init__()
        self.decoder_cfg = decoder_cfg
        depth = decoder_cfg.NUM_LAYERS
        dpr = [x.item() for x in torch.linspace(0, decoder_cfg.DROP_PATH_RATE, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_cfg.NUM_FEATURES, num_heads=decoder_cfg.NUM_HEADS, mlp_hidden_dim=decoder_cfg.NUM_HIDDEN_FEATURES, 
                qkv_bias=False, qk_scale=False, drop=0., attn_drop=0., drop_path = dpr[i]
                )
            for i in range(depth)])

        self.norm = nn.LayerNorm(decoder_cfg.NUM_FEATURES)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        x = x + pos
        for _, block in enumerate(self.blocks):
            x = block(x)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x

class MaskTransformer(nn.Module):
    def __init__(self, mae_cfg, pos_encoder):
        super().__init__()
        self.mae_cfg = mae_cfg
        self.blocks = TransformerEncoder(
            self.mae_cfg.ENCODER,
            pos_encoder
        )

        self.norm = nn.LayerNorm(self.mae_cfg.ENCODER.NUM_FEATURES)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def _mask_points_rand(self, points, no_aug = False):
        '''
            Args:
                points: (B, G, C)
            Returns:
                mask (bool): (B, G) 
        '''
        B, G, _ = points.shape
        if no_aug or self.mae_cfg.MASK_RATIO == 0:
            return torch.zeros(points.shape[:2]).bool()
        
        self.num_mask = int(self.mae_cfg.MASK_RATIO * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        
        return overall_mask.to(points.device) # B G

    def forward(self, pooled_features, positional_input, noaug = False):
        # generate mask
        bool_masked_pos = self._mask_grid_rand(positional_input)

        batch_size, seq_len, C = pooled_features.size()

        x_vis = pooled_features[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        pos = positional_input[~bool_masked_pos].reshape(batch_size, -1, 3)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class FrequencyPositionalEncoding3d(nn.Module):
    def __init__(self, d_model, max_spatial_shape, dropout=0.1):
        """
        Sine + Cosine positional encoding based on Attention is all you need (https://arxiv.org/abs/1706.03762) in 3D. Using the same concept as DETR,
        the sinusoidal encoding is independent across each spatial dimension.

        Args:
            d_model: Dimension of the input features. Must be divisible by 6 ((cos + sin) * 3 dimensions = 6)
            max_spatial_shape: (3,) Size of each dimension
            dropout: Dropout probability
        """
        super().__init__()

        assert len(max_spatial_shape) == 3, 'Spatial dimension must be 3'
        assert d_model % 6 == 0, f'Feature dimension {d_model} not divisible by 6'
        self.max_spatial_shape = max_spatial_shape

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([d_model] + list(max_spatial_shape))

        d_model = int(d_model / len(max_spatial_shape))

        # Equivalent to attention is all you need encoding: https://arxiv.org/abs/1706.03762
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        pos_x = torch.arange(0., max_spatial_shape[0]).unsqueeze(1)
        pos_y = torch.arange(0., max_spatial_shape[1]).unsqueeze(1)
        pos_z = torch.arange(0., max_spatial_shape[2]).unsqueeze(1)

        pe[0:d_model:2, ...] = torch.sin(pos_x * div_term).transpose(0, 1)[:, :, None, None].repeat(1, 1, max_spatial_shape[1], max_spatial_shape[2])
        pe[1:d_model:2, ...] = torch.cos(pos_x * div_term).transpose(0, 1)[:, :, None, None].repeat(1, 1, max_spatial_shape[1], max_spatial_shape[2])
        pe[d_model:2*d_model:2, ...] = torch.sin(pos_y * div_term).transpose(0, 1)[:, None, :, None].repeat(1, max_spatial_shape[0], 1, max_spatial_shape[2])
        pe[d_model+1:2*d_model:2, ...] = torch.cos(pos_y * div_term).transpose(0, 1)[:, None, :, None].repeat(1, max_spatial_shape[0], 1, max_spatial_shape[2])
        pe[2*d_model:3*d_model:2, ...] = torch.sin(pos_z * div_term).transpose(0, 1)[:, None, None, :].repeat(1, max_spatial_shape[0], max_spatial_shape[1], 1)
        pe[2*d_model+1:3*d_model:2, ...] = torch.cos(pos_z * div_term).transpose(0, 1)[:, None, None, :].repeat(1, max_spatial_shape[0], max_spatial_shape[1], 1)

        self.register_buffer('pe', pe)

    def forward(self, point_features, positional_input, grid_size=None):
        """
        Args:
            point_features: (b, xyz, f)
            positional_input: (b, xyz, 3)

        Returns:
            point_features: (b, xyz, f)
        """
        assert len(point_features.shape) == 3
        num_points = point_features.shape[1]
        num_features = point_features.shape[2]
        if grid_size == None:
            grid_size = self.max_spatial_shape
        assert num_points == grid_size.prod()

        pe =  self.pe[:, :grid_size[0], :grid_size[1], :grid_size[2]].permute(1, 2, 3, 0).contiguous().view(1, num_points, num_features)
        point_features = point_features + pe
        return self.dropout(point_features)


class FeedForwardPositionalEncoding(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_input, d_output // 2, 1),
            nn.BatchNorm1d(d_output // 2),
            nn.ReLU(d_output // 2),
            nn.Conv1d(d_output // 2, d_output, 1),
        )

    def forward(self, positional_input, grid_size=None):
        """
        Args:
            local_point_locations: (b, xyz, l)

        Returns:
            point_features: (b, xyz, f)
        """
        pos_encoding = self.ffn(positional_input.permute(0, 2, 1))
        return pos_encoding.permute(0, 2, 1)


def get_positional_encoder(attention_cfg):
    pos_encoder = None
    if attention_cfg.POSITIONAL_ENCODER == 'frequency':
        pos_encoder = FrequencyPositionalEncoding3d(d_model=attention_cfg.NUM_FEATURES,
                                                    max_spatial_shape=torch.IntTensor([pool_cfg.GRID_SIZE] * 3),
                                                    dropout=attention_cfg.DROPOUT)
    elif attention_cfg.POSITIONAL_ENCODER == 'grid_points':
        pos_encoder = FeedForwardPositionalEncoding(d_input=3, d_output=attention_cfg.ENCODER.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'grid_points_corners':
        pos_encoder = FeedForwardPositionalEncoding(d_input=27, d_output=attention_cfg.ENCODER.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density':
        pos_encoder = FeedForwardPositionalEncoding(d_input=1, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density_grid_points':
        pos_encoder = FeedForwardPositionalEncoding(d_input=4, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density_centroid':
        pos_encoder = FeedForwardPositionalEncoding(d_input=7, d_output=attention_cfg.NUM_FEATURES)
    
    return pos_encoder