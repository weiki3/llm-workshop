from datetime import datetime
import time
from datasets import load_from_disk
import tiktoken
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn

BOS_IDX = 50256
EOS_IDX = 50257
PAD_IDX = 50258

BLOCK_SIZE = 256
BATCH_SIZE = 200
STEP_NUM = 100000


@dataclass
class TransformerConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = 50259
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8
    p_drop: float = 0.1
    eps_ls: float = 0.1


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        assert config.n_embd % config.n_head == 0, f'n_embd should be multiple n_head'

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.c_proj(output)
        return output


class CausalAttention(Attention):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.c_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        assert config.n_embd % config.n_head == 0, f'n_embd should be multiple n_head'

        self.c_attn_x = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn_y = nn.Linear(config.n_embd, config.n_embd * 2)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x, y):
        B, T, C = x.size()
        _, S, _ = y.size()

        q = self.c_attn_x(x)  # (B, T, C)
        kv = self.c_attn_y(y)  # (B, S, 2 * C)
        k, v = kv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.c_proj(output)
        return output


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.attn = Attention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x):
        x = x + self.dropout(self.attn(x))
        x = self.ln_1(x)
        x = x + self.dropout(self.mlp(x))
        x = self.ln_2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.attn = CausalAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x, y):
        x = x + self.dropout(self.attn(x))
        x = self.ln_1(x)
        x = x + self.dropout(self.cross_attn(x, y))
        x = self.ln_2(x)
        x = x + self.dropout(self.mlp(x))
        x = self.ln_3(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h_enc=nn.ModuleList([EncoderBlock(config)
                                for _ in range(config.n_layer)]),
            h_dec=nn.ModuleList([DecoderBlock(config)
                                for _ in range(config.n_layer)])
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.dropout = nn.Dropout(config.p_drop)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # if hasattr(module, 'SCALE_INIT'):
            #     std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src_idx, idx, target_idx=None):
        B, T = idx.size()
        src_B, S = src_idx.size()
        assert B == src_B, f"src batch size {src_B} must match target batch size {B}"

        pos = torch.arange(0, T, 1, dtype=torch.long, device=idx.device)
        src_pos = torch.arange(0, S, 1, dtype=torch.long,
                               device=src_idx.device)
        pos_embd = self.transformer.wpe(pos)
        src_pos_embd = self.transformer.wpe(src_pos)
        tok_embd = self.transformer.wte(idx)
        src_tok_embd = self.transformer.wte(src_idx)

        x = pos_embd + tok_embd
        x = self.dropout(x)

        src_y = src_pos_embd + src_tok_embd
        src_y = self.dropout(src_y)

        for block in self.transformer.h_enc:
            src_y = block(src_y)
        for block in self.transformer.h_dec:
            x = block(x, src_y)

        logits = self.lm_head(x)
        loss = None
        if target_idx is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_idx.reshape(-1),
                ignore_index=PAD_IDX,
                label_smoothing=self.config.eps_ls)

        return logits, loss


gpt2_enc = tiktoken.get_encoding("gpt2")
enc = tiktoken.Encoding(
    name="mygpt2",
    pat_str=gpt2_enc._pat_str,
    mergeable_ranks=gpt2_enc._mergeable_ranks,
    special_tokens={
        "<BOS>": 50256,
        "<EOS>": 50257,
        "<PAD>": 50258
    }
)


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        self.ds = load_from_disk('./my_wmt14_tokens')
        self.length = len(self.ds)

        self.curr_pos = 0

        print(f"load tokens {self.length * 2 * T}")
        print(f"1 epoch = {self.length // B} batches")

    def next_batch(self):
        B, T = self.B, self.T

        if self.curr_pos > self.length - B:
            self.curr_pos = 0

        src_idx = self.ds['de'][self.curr_pos:self.curr_pos + B]
        idx = self.ds['en'][self.curr_pos:self.curr_pos + B]

        for i in range(B):
            src_idx[i] = src_idx[i][:T - 1] + [EOS_IDX]
            if len(src_idx[i]) < T:
                src_idx[i] += [PAD_IDX] * (T - len(src_idx[i]))

            idx[i] = idx[i][:T - 1] + [EOS_IDX]
            if len(idx[i]) < T:
                idx[i] += [PAD_IDX] * (T - len(idx[i]))
            idx[i] = [BOS_IDX] + idx[i]

        src_idx = torch.tensor(src_idx).view(B, T)
        idx = torch.tensor(idx).view(B, T + 1)

        self.curr_pos += B
        return src_idx, idx[:, :-1], idx[:, 1:]


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')

device = 'cuda:3'

train_loader = DataLoaderLite(BATCH_SIZE, BLOCK_SIZE)
model = Transformer(TransformerConfig())
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
for _ in range(STEP_NUM):
    t0 = time.time()

    src_idx, idx, target_idx = train_loader.next_batch()
    src_idx, idx, target_idx = src_idx.to(
        device), idx.to(device), target_idx.to(device)

    optimizer.zero_grad()
    with torch.autocast(device, dtype=torch.bfloat16):
        logits, loss = model(src_idx, idx, target_idx)

    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f'loss {loss:.2f} dt {dt:.2f}')

torch.save(model.state_dict(), f'./{datetime.now().strftime("%Y%m%d_%H%M")}.model')
