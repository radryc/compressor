#!/usr/bin/env python3
"""GPU/CPU trainer for the nzip byte-level GRU.

The exported NGRM artifact must match the Go codec exactly:

    in_proj[b] = w_in.weight[b] + b_in          (256 x 3H, input bias folded)
    w_hh       = recurrent weights              (3H x H, gates r, z, n)
    b_hh       = recurrent bias                 (3H)
    w_bit      = bit-head weights               (8 x H)
    b_bit      = bit-head bias                  (8)

The recurrence is written out explicitly instead of using nn.GRU so that the
gate order, the folded input projection and the candidate formula
tanh(in_n + r * (W_hn h + b_hn)) match the Go implementation.

The model predicts each of the eight bits of the *next* byte, so the loss is
binary cross-entropy summed over bit positions. The 256-way softmax of the first
generation is gone: it cost 256*H multiplies per byte, this costs 8*H.

Usage:
    python train.py --data otel.ndjson --out models/101.bin --model-id 101 \
        --hidden 32 --epochs 6 --seq-len 64
"""

import argparse
import math
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_MAGIC = b"NGRM"
MODEL_VERSION = 2
MAX_HIDDEN = 96


class ByteGRU(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        if not 1 <= hidden_dim <= MAX_HIDDEN:
            raise ValueError(f"hidden_dim must be 1..{MAX_HIDDEN}")
        self.hidden_dim = hidden_dim
        h = hidden_dim
        self.w_in = nn.Embedding(256, 3 * h)
        self.b_in = nn.Parameter(torch.zeros(3 * h))
        self.w_hh = nn.Parameter(torch.empty(3 * h, h))
        self.b_hh = nn.Parameter(torch.zeros(3 * h))
        self.w_bit = nn.Parameter(torch.empty(8, h))
        self.b_bit = nn.Parameter(torch.zeros(8))
        scale = 1.0 / math.sqrt(h)
        for p in (self.w_in.weight, self.w_hh, self.w_bit):
            nn.init.uniform_(p, -scale, scale)
        self._h = h

    def forward(self, x, h=None):
        """x: (B, T) int64 previous bytes. Returns bit logits (B, T, 8), h."""
        hdim = self._h
        b, t = x.shape
        if h is None:
            h = x.new_zeros((b, hdim), dtype=torch.float32)
        outs = []
        pre_in = self.w_in(x) + self.b_in  # (B,T,3H)
        for step in range(t):
            pre = pre_in[:, step]  # (B,3H)
            r = torch.sigmoid(pre[:, :hdim] + h @ self.w_hh[:hdim].T + self.b_hh[:hdim])
            z = torch.sigmoid(
                pre[:, hdim : 2 * hdim] + h @ self.w_hh[hdim : 2 * hdim].T + self.b_hh[hdim : 2 * hdim]
            )
            hn = h @ self.w_hh[2 * hdim :].T + self.b_hh[2 * hdim :]
            n = torch.tanh(pre[:, 2 * hdim :] + r * hn)
            h = (1 - z) * n + z * h
            outs.append(h @ self.w_bit.T + self.b_bit)
        return torch.stack(outs, dim=1), h

    def step(self, prev, h=None):
        logits, h = self.forward(prev.view(1, 1), h)
        return logits[:, 0, :], h


def export_model(model: "ByteGRU", model_id: int, out_path: str):
    model.eval().cpu()
    with torch.no_grad():
        in_proj = (model.w_in.weight + model.b_in).contiguous()  # (256, 3H)
        tensors = (
            in_proj,
            model.w_hh.contiguous(),
            model.b_hh.contiguous(),
            model.w_bit.contiguous(),
            model.b_bit.contiguous(),
        )
    floats = sum(t.numel() for t in tensors)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sHHI", MODEL_MAGIC, MODEL_VERSION, model.hidden_dim, model_id))
        f.write(struct.pack("<I", 0))  # reserved
        for t in tensors:
            f.write(t.numpy().astype("<f4").tobytes())
    size_kb = (16 + floats * 4) / 1024
    print(f"exported model_id={model_id} H={model.hidden_dim} -> {out_path} ({size_kb:.1f} KiB)")


def make_batch(data, starts, seq_len):
    """inputs[t] is the byte before targets[t]; position 0 uses a 0 start token."""
    idx = torch.arange(seq_len)
    tgt_idx = starts.unsqueeze(1) + idx  # (B,T)
    x = torch.zeros_like(tgt_idx)
    x[:, 1:] = data[tgt_idx[:, :-1]]
    return x, data[tgt_idx]


def bits_of(targets):
    shifts = torch.arange(7, -1, -1)
    return ((targets.unsqueeze(-1) >> shifts) & 1).float()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    raw = np.fromfile(args.data, dtype=np.uint8)
    if len(raw) < args.seq_len + 8:
        raise ValueError(f"dataset too small ({len(raw)} bytes)")

    split = int(len(raw) * (1 - args.holdout))
    train_np, eval_np = raw[:split], raw[split:]
    data = torch.from_numpy(train_np.astype(np.int64))
    model = ByteGRU(hidden_dim=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    windows = len(data) - args.seq_len - 1
    per_epoch = min(windows, args.steps_per_epoch)
    print(
        f"device={device} hidden={args.hidden} bytes={len(data)} "
        f"windows/epoch={per_epoch} bits-parameter={sum(p.numel() for p in model.parameters())}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        order = torch.randperm(windows)
        for i in range(0, per_epoch, args.batch_size):
            starts = order[i : i + args.batch_size]
            x, y = make_batch(data, starts, args.seq_len)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, bits_of(y))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(starts)
            seen += len(starts)
        bpb = (total / max(seen, 1)) * 8  # mean BCE per bit, times 8 bits/byte
        print(f"epoch {epoch:02d}/{args.epochs}  train {bpb:.3f} bits/byte")

    if len(eval_np) > args.seq_len + 2:
        model.eval()
        ev = torch.from_numpy(eval_np.astype(np.int64))
        starts = torch.arange(0, len(ev) - args.seq_len - 1, args.seq_len)
        total, seen = 0.0, 0
        with torch.no_grad():
            for s in starts:
                idx = s + torch.arange(args.seq_len)
                x = ev[idx - 1].unsqueeze(0)
                x[:, 0] = 0
                y = ev[idx].unsqueeze(0)
                logits, _ = model(x.to(device))
                loss = F.binary_cross_entropy_with_logits(logits, bits_of(y.to(device)))
                total += loss.item() * args.seq_len
                seen += args.seq_len
        print(f"holdout {total / max(seen, 1) * 8:.3f} bits/byte (float model)")
        print("verify with the real codec before promoting: nzip eval -model <out> -in <holdout>")

    export_model(model, args.model_id, args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="training corpus (raw bytes)")
    p.add_argument("--out", required=True, help="output NGRM artifact")
    p.add_argument("--model-id", type=int, required=True)
    p.add_argument("--hidden", type=int, default=32, help=f"hidden size (1..{MAX_HIDDEN})")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--holdout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1)
    train(p.parse_args())
