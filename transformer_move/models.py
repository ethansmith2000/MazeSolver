
import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, bias=False):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.h = heads
        self.dh = dim // heads

    def forward(self, x, mask=None):
        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], self.h, self.dh).transpose(1, 2), (self.to_qkv(x).chunk(3, dim=-1)))
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True).transpose(1, 2).reshape(q.shape[0], -1, self.h * self.dh)
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
                 ada_norm=False
                 ):

        super().__init__()
        norm_class = AdaNorm if ada_norm else nn.LayerNorm
        self.self_attn = SelfAttention(query_dim, heads=heads, dropout=dropout)
        self.self_norm = norm_class(query_dim)

        self.cross_attn = CrossAttention(query_dim, context_dim, heads=heads, dropout=dropout) if use_cross_attn else identity
        self.cross_norm = norm_class(query_dim) if use_cross_attn else identity

        self.ff = FeedForward(query_dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = norm_class(query_dim)

        self.gradient_checkpointing = False

    def forward(self, x, context, ada_emb=None, attn_mask=None, cross_attn_mask=None):
        if self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(self.self_attn, self.self_norm(x), attn_mask) + x
            x = torch.utils.checkpoint.checkpoint(self.cross_attn, self.cross_norm(x), context, cross_attn_mask) + x
            x = torch.utils.checkpoint.checkpoint(self.ff, self.ff_norm(x)) + x
        else:
            x = self.self_attn(self.self_norm(x), attn_mask) + x
            x = self.cross_attn(self.cross_norm(x), context, cross_attn_mask) + x
            x = self.ff(self.ff_norm(x)) + x

        return x


class Transformer(nn.Module):

    def __init__(self, 
                    num_layers_encoder=6, 
                    num_layers_decoder=8, 
                    dim=512, 
                    heads=8, 
                    ff_mult=4, 
                    maze_size=20, 
                    movements=True
                    ):

        super().__init__()
        self.maze_one_side = maze_size * 2 + 1
        self.total_pixs = self.maze_one_side * self.maze_one_side

        self.num_options = 6 if movements else self.total_pixs

        # 0=down, 1=up, 2=right, 3=left
        self.embed_path = nn.Embedding(self.num_options, dim)
        # 0=empty, 1=wall, 2=start, 3=end
        self.embed_maze = nn.Embedding(4, dim)


        self.pos_embs_encoder = nn.Parameter(torch.randn(1, self.total_pixs, dim) * 0.01)
        self.pos_embs_decoder = nn.Parameter(torch.randn(1, self.total_pixs, dim) * 0.01)

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
        self.out_proj = nn.Linear(dim, self.num_options)


    def enable_gradient_checkpointing(self):
        for layer in self.encoder:
            layer.gradient_checkpointing = True
        for layer in self.decoder:
            layer.gradient_checkpointing = True


    def forward(self, path, maze, attn_mask=None):
        # maze goes through encoder
        b, _, h, w = maze.shape
        maze = maze.squeeze(1).reshape(b, -1)
        maze_embs = self.embed_maze(maze)
        maze_embs = maze_embs + self.pos_embs_encoder.repeat(b, 1, 1)

        for layer in self.encoder:
            maze_embs = layer(maze_embs, None)

        # path should already be flattened and padded to fit maze
        b, s = path.shape #b, s
        path = self.embed_path(path)
        path = path + self.pos_embs_decoder[:, :s, :].repeat(b, 1, 1)
        for layer in self.decoder:
            path = layer(path, maze_embs, attn_mask=attn_mask)
        path = self.final_layer_norm(path)
        path = self.out_proj(path)

        return path