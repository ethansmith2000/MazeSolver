
import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, bias=False):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.h = heads
        self.dh = dim // heads

    def forward(self, x, mask=None):
        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], self.h, self.dh).transpose(1, 2), (self.to_qkv(x).chunk(3, dim=-1)))
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(q.shape[0], -1, self.h * self.dh)
        return attn_output


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dropout=0.0, bias=False):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=bias)
        self.h = heads
        self.dh = dim // heads

    def forward(self, x, context, mask=None):
        q = self.to_q(x).reshape(x.shape[0], -1, self.h, self.dh).transpose(1, 2)
        k, v = map(lambda t: t.reshape(t.shape[0], -1, self.h, self.dh).transpose(1, 2), (self.to_kv(context).chunk(2, dim=-1)))
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(q.shape[0], -1, self.h * self.dh)
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, bias=True, act_fn=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult, bias=bias),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AdaNorm(torch.nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.act_fn = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, time_emb):
        time_emb = self.act_fn(time_emb)
        scale, shift = self.linear(time_emb).chunk(2, dim=-1)
        return self.norm(x) * scale[:, None, :] + shift[:, None, :]

def identity(x, *args, **kwargs):
    return x

class TransformerLayer(nn.Module):

    def __init__(self, 
                query_dim=768,
                 context_dim=1024,
                 heads=8, 
                 dropout=0.0,
                 ff_mult=4,
                 use_cross_attn=True,
                 ):

        super().__init__()
        self.self_attn = SelfAttention(query_dim, heads=heads, dropout=dropout)
        self.self_norm = AdaNorm(query_dim)

        self.cross_attn = CrossAttention(query_dim, context_dim, heads=heads, dropout=dropout) if use_cross_attn else identity
        self.cross_norm = AdaNorm(query_dim) if use_cross_attn else identity

        self.ff = FeedForward(query_dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = AdaNorm(query_dim)

        self.gradient_checkpointing = False

    def forward(self, x, context, ada_emb=None, attn_mask=None, cross_attn_mask=None):
        if self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.self_attn, self.self_norm(x, ada_emb), attn_mask) + x
            x = torch.utils.checkpoint.checkpoint(self.cross_attn, self.cross_norm(x, ada_emb), context, cross_attn_mask) + x
            x = torch.utils.checkpoint.checkpoint(self.ff, self.ff_norm(x, ada_emb)) + x
        else:
            x = self.self_attn(self.self_norm(x, ada_emb), attn_mask) + x
            x = self.cross_attn(self.cross_norm(x, ada_emb), context, cross_attn_mask) + x
            x = self.ff(self.ff_norm(x, ada_emb)) + x

        return x


class FourierEmbedder:
    def __init__(self, num_freqs, temperature):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)



class DiffusionTransformer(nn.Module):

    def __init__(self, in_channels=3, 
                    out_channels=3, 
                    num_layers_encoder=6, 
                    num_layers_decoder=8, 
                    dim=512, 
                    heads=8, 
                    ff_mult=4, 
                    maze_size=20, 
                    act_fn="silu", 
                    num_freqs=64
                    ):

        super().__init__()
        # all coords normalized -1, 1
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, temperature=60)
        self.embed_path = nn.Linear(num_freqs * 2 * in_channels, dim)

        # 0=empty, 1=wall, 2=start, 3=end
        self.embed_maze = nn.Embedding(4, dim)
        self.maze_one_side = maze_size * 2 + 1
        self.total_pixs = self.maze_one_side * self.maze_one_side

        self.pos_embs_encoder = nn.Parameter(torch.randn(1, self.total_pixs, dim) * 0.01)
        self.pos_embs_decoder = nn.Parameter(torch.randn(1, self.total_pixs, dim) * 0.01)

        self.time_proj = Timesteps(dim//2, flip_sin_to_cos=False, downscale_freq_shift=0.0)
        self.time_embedding = TimestepEmbedding(
            dim//2,
            dim,
            act_fn=act_fn,
            post_act_fn=None,
            cond_proj_dim=None,
        )

        self.encoder = nn.ModuleList([TransformerLayer(query_dim=dim,
                                                        context_dim=dim,
                                                        heads=heads,
                                                        dropout=0.0,
                                                        ff_mult=ff_mult,
                                                        use_cross_attn=False,
                                                        ) for _ in range(num_layers_encoder)])


        self.decoder = nn.ModuleList([TransformerLayer(query_dim=dim,
                                                        context_dim=dim,
                                                        heads=heads,
                                                        dropout=0.0,
                                                        ff_mult=ff_mult,
                                                        use_cross_attn=True,
                                                        ) for _ in range(num_layers_decoder)])

        self.final_layer_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_channels)


    def enable_gradient_checkpointing(self):
        for layer in self.encoder:
            layer.gradient_checkpointing = True
        for layer in self.decoder:
            layer.gradient_checkpointing = True


    def forward(self, path, maze, timesteps, attn_mask=None, dropout_mask=None):
        #timestep
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(path.device)
        timesteps = timesteps.expand(path.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=path.dtype)
        emb = self.time_embedding(t_emb, None)

        # maze goes through encoder
        b, _, h, w = maze.shape
        maze = maze.squeeze(1).reshape(b, -1)
        maze_embs = self.embed_maze(maze)
        maze_embs = maze_embs + self.pos_embs_encoder.repeat(b, 1, 1)

        for layer in self.encoder:
            maze_embs = layer(maze_embs, None, emb)

        if dropout_mask is not None:
            maze_embs = maze_embs * dropout_mask[:, None, None]

        # path should already be flattened and padded to fit maze
        b, s, xy = path.shape #b, s, 3
        path = self.fourier_embedder(path)
        path = self.embed_path(path)
        path = path + self.pos_embs_decoder.repeat(b, 1, 1)
        for layer in self.decoder:
            path = layer(path, maze_embs, emb, attn_mask=attn_mask)
        path = self.final_layer_norm(path)
        path = self.out_proj(path)

        return path