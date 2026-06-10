import argparse
from datetime import datetime
import os
from pathlib import Path
import time
from datasets import load_from_disk
import tiktoken
import torch
from torch.nn import functional as F
from dataclasses import dataclass
import torch.nn as nn

BOS_IDX = 50256
EOS_IDX = 50257
PAD_IDX = 50258

BLOCK_SIZE = 256
BATCH_SIZE = 10
STEP_NUM = 100000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class TransformerConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = 50259
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8
    p_drop: float = 0.1
    eps_ls: float = 0.1
    pad_idx: int = PAD_IDX


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        assert config.n_embd % config.n_head == 0, f'n_embd should be multiple n_head'

        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.config.p_drop if self.training else 0.0,
        )
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

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.config.p_drop if self.training else 0.0,
        )
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
        self.c_proj.SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.n_head = config.n_head

    def forward(self, x, y, attn_mask=None):
        B, T, C = x.size()
        _, S, _ = y.size()

        q = self.c_attn_x(x)  # (B, T, C)
        kv = self.c_attn_y(y)  # (B, S, 2 * C)
        k, v = kv.split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.config.p_drop if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.c_proj(output)
        return output


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU('tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
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

    def forward(self, x, attn_mask=None):
        x = x + self.dropout(self.attn(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
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

    def forward(self, x, y, cross_attn_mask=None):
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.cross_attn(self.ln_2(x), y, attn_mask=cross_attn_mask))
        x = x + self.dropout(self.mlp(self.ln_3(x)))
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
                                for _ in range(config.n_layer)]),
            ln_enc_f=nn.LayerNorm(config.n_embd),
            ln_dec_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.dropout = nn.Dropout(config.p_drop)
        self.register_buffer(
            "position_ids",
            torch.arange(config.block_size, dtype=torch.long),
            persistent=False,
        )

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src_idx, idx, target_idx=None):
        B, T = idx.size()
        src_B, S = src_idx.size()
        assert B == src_B, f"src batch size {src_B} must match target batch size {B}"
        assert T <= self.config.block_size, f"T {T} exceeds block size {self.config.block_size}"
        assert S <= self.config.block_size, f"S {S} exceeds block size {self.config.block_size}"
        src_attn_mask = src_idx.ne(self.config.pad_idx).view(B, 1, 1, S)

        pos = self.position_ids[:T]
        src_pos = self.position_ids[:S]
        pos_embd = self.transformer.wpe(pos)
        src_pos_embd = self.transformer.wpe(src_pos)
        tok_embd = self.transformer.wte(idx)
        src_tok_embd = self.transformer.wte(src_idx)

        x = pos_embd + tok_embd
        x = self.dropout(x)

        src_y = src_pos_embd + src_tok_embd
        src_y = self.dropout(src_y)

        for block in self.transformer.h_enc:
            src_y = block(src_y, attn_mask=src_attn_mask)
        src_y = self.transformer.ln_enc_f(src_y)
        for block in self.transformer.h_dec:
            x = block(x, src_y, cross_attn_mask=src_attn_mask)
        x = self.transformer.ln_dec_f(x)

        logits = self.lm_head(x)
        loss = None
        if target_idx is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_idx.reshape(-1),
                ignore_index=self.config.pad_idx,
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
    def __init__(self, B, T, dataset_path=None, shuffle=True, seed=42):
        self.B = B
        self.T = T
        self.shuffle = shuffle
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        dataset_path = Path(dataset_path) if dataset_path else SCRIPT_DIR / 'my_wmt14_tokens'
        self.ds = load_from_disk(str(dataset_path))
        self.length = len(self.ds)

        self.curr_pos = 0

        print(f"load tokens {self.length * 2 * T}")
        print(f"1 epoch = {self.length // B} batches")
        print(f"shuffle = {self.shuffle}")

    def next_batch(self):
        B, T = self.B, self.T

        if self.shuffle:
            batch_idx = torch.randint(
                low=0,
                high=self.length,
                size=(B,),
                generator=self.generator,
            ).tolist()
            batch = self.ds[batch_idx]
        else:
            if self.curr_pos > self.length - B:
                self.curr_pos = 0
            batch = self.ds[self.curr_pos:self.curr_pos + B]
            self.curr_pos += B

        src_batch = []
        idx_batch = []
        target_batch = []

        for src_tokens, target_tokens in zip(batch['de'], batch['en']):
            src = src_tokens[:T - 1] + [EOS_IDX]
            src += [PAD_IDX] * (T - len(src))

            target = target_tokens[:T - 1] + [EOS_IDX]
            target += [PAD_IDX] * (T - len(target))

            src_batch.append(src)
            idx_batch.append([BOS_IDX] + target[:-1])
            target_batch.append(target)

        src_idx = torch.tensor(src_batch, dtype=torch.long)
        idx = torch.tensor(idx_batch, dtype=torch.long)
        target_idx = torch.tensor(target_batch, dtype=torch.long)

        return src_idx, idx, target_idx


def get_train_device(requested=None):
    requested = requested or os.environ.get('TRAIN_DEVICE')
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda:3' if torch.cuda.device_count() > 3 else 'cuda')
    return torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Train the toy encoder-decoder transformer.")
    parser.add_argument("--steps", type=int, default=STEP_NUM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--n-embd", type=int, default=TransformerConfig.n_embd)
    parser.add_argument("--n-layer", type=int, default=TransformerConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=TransformerConfig.n_head)
    parser.add_argument("--dropout", type=float, default=TransformerConfig.p_drop)
    parser.add_argument("--label-smoothing", type=float, default=TransformerConfig.eps_ls)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_optimizer(model, lr, weight_decay, device_type):
    decay_params = []
    nodecay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer_kwargs = dict(lr=lr, betas=(0.9, 0.98), eps=1e-9)
    if device_type == 'cuda':
        optimizer_kwargs['fused'] = True
    return torch.optim.AdamW(optimizer_groups, **optimizer_kwargs)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('high')

    device = get_train_device(args.device)
    device_type = device.type

    train_loader = DataLoaderLite(
        args.batch_size,
        args.block_size,
        dataset_path=args.dataset_path,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    config = TransformerConfig(
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        p_drop=args.dropout,
        eps_ls=args.label_smoothing,
    )
    model = Transformer(config)
    model.to(device)
    if args.compile and hasattr(model, "compile"):
        model.compile()

    optimizer = build_optimizer(model, args.lr, args.weight_decay, device_type)
    ema_loss = None
    for step in range(1, args.steps + 1):
        t0 = time.time()

        src_idx, idx, target_idx = train_loader.next_batch()
        src_idx = src_idx.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        target_idx = target_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == 'cuda'):
            _, loss = model(src_idx, idx, target_idx)

        loss.backward()
        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if device_type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        loss_item = loss.detach().item()
        ema_loss = loss_item if ema_loss is None else 0.9 * ema_loss + 0.1 * loss_item
        valid_tokens = target_idx.ne(config.pad_idx).sum().item()
        tokens_per_sec = valid_tokens / (dt / 1000)
        should_log = step == 1 or step == args.steps or step % args.log_interval == 0
        if should_log:
            grad_text = "" if grad_norm is None else f" grad_norm {float(grad_norm):.2f}"
            print(
                f"step {step:5d}/{args.steps} loss {loss_item:.4f} "
                f"ema_loss {ema_loss:.4f}{grad_text} dt {dt:.2f}ms tok/s {tokens_per_sec:.0f}"
            )

    if args.save:
        model_path = SCRIPT_DIR / f'{datetime.now().strftime("%Y%m%d_%H%M")}.model'
        torch.save(model.state_dict(), model_path)
        print(f"saved {model_path}")


if __name__ == '__main__':
    main()
