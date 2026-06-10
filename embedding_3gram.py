import argparse
import math
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OMIT = frozenset("qx")
SENTINEL = "^"

def normalize(raw, omit=OMIT):
    out = []
    for ch in raw:
        if ch in (" ", "\n"):
            out.append(" ")
        elif ch.isalpha():
            low = ch.lower()
            if low not in omit:
                out.append(low)
    return out

def build_vocab(stream):
    alphabet = sorted(set(stream))
    if SENTINEL in alphabet:
        alphabet.remove(SENTINEL)
    alphabet = [SENTINEL] + alphabet
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return alphabet, char_to_idx, idx_to_char

def build_ngram_counts(stream, max_order):
    counts = {}
    for order in range(1, max_order + 1):
        counts[order] = defaultdict(lambda: defaultdict(int))
    for i in range(len(stream) - 1):
        nxt = stream[i + 1]
        for order in range(1, max_order + 1):
            start = i - order + 1
            if start < 0:
                continue
            ctx = tuple(stream[start:i + 1])
            counts[order][ctx][nxt] += 1
    return counts

def build_global_freq(stream, alphabet):
    freq = defaultdict(int)
    for ch in stream:
        if ch in alphabet and ch != SENTINEL:
            freq[ch] += 1
    total = sum(freq.values()) or 1
    return sorted([ch for ch in alphabet if ch != SENTINEL], key=lambda c: -freq[c] / total)

def make_oracle(counts, global_order, max_order):
    def reorder(context):
        for order in range(min(max_order, len(context)), 0, -1):
            ctx_key = tuple(context[-order:])
            if ctx_key in counts[order]:
                followers = counts[order][ctx_key]
                ranked = sorted(followers.keys(), key=lambda c: -followers[c])
                ranked_set = set(ranked)
                tail = [c for c in global_order if c not in ranked_set]
                return ranked + tail
        return global_order
    return reorder

def static_reorder(global_order):
    def reorder(context):
        return global_order
    return reorder

def simulate(stream, alphabet, reorder_fn, context_len):
    total_cost = 0
    total_chars = 0
    padding = [SENTINEL] * context_len
    padded = padding + stream
    candidate_set = [ch for ch in alphabet if ch != SENTINEL]
    for i in range(context_len, len(padded)):
        ch = padded[i]
        if ch not in candidate_set:
            continue
        context = padded[i - context_len:i]
        ordering = reorder_fn(context)
        try:
            pos = ordering.index(ch)
        except ValueError:
            pos = len(ordering)
        total_cost += pos
        total_chars += 1
    avg = total_cost / total_chars if total_chars else 0.0
    return total_cost, total_chars, avg

def make_examples(stream, context_len, char_to_idx):
    padded = [SENTINEL] * context_len + stream
    x = []
    y = []
    for i in range(context_len, len(padded)):
        tgt = padded[i]
        if tgt not in char_to_idx:
            continue
        ctx = padded[i - context_len:i]
        if any(ch not in char_to_idx for ch in ctx):
            continue
        x.append([char_to_idx[ch] for ch in ctx])
        y.append(char_to_idx[tgt])
    if not x:
        raise ValueError("No training examples were created from the input text.")
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class ContextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, dim, context_len, sentinel_idx):
        super().__init__()
        self.context_len = context_len
        self.sentinel_idx = sentinel_idx
        self.context_embs = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(context_len)])
        self.output_emb = nn.Embedding(vocab_size, dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        state = None
        for pos, emb in enumerate(self.context_embs):
            part = emb(x[:, pos])
            state = part if state is None else state + part
        logits = state @ self.output_emb.weight.t() + self.output_bias
        logits = logits.clone()
        logits[:, self.sentinel_idx] = -1e9
        return logits

    def predict_ordering(self, context_ids, idx_to_char):
        device = self.output_bias.device
        x = torch.tensor(context_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.forward(x).squeeze(0)
            order = torch.argsort(logits, descending=True).tolist()
        return [idx_to_char[i] for i in order if i != self.sentinel_idx]

    def parameter_bytes(self):
        return sum(p.numel() for p in self.parameters()) * 4

def train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, generator=generator)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_seen += xb.size(0)
        train_loss = total_loss / max(total_seen, 1)
        val_loss = evaluate_loss(model, val_x, val_y, batch_size)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_loss(model, x, y, batch_size):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            total_loss += loss.item() * xb.size(0)
            total_seen += xb.size(0)
    return total_loss / max(total_seen, 1)

def evaluate_rotations(model, x, y, idx_to_char, char_to_idx, batch_size):
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    valid_indices = [i for i in range(len(idx_to_char)) if i != char_to_idx[SENTINEL]]
    valid_indices_t = torch.tensor(valid_indices, dtype=torch.long)
    full_to_valid = torch.full((len(idx_to_char),), -1, dtype=torch.long)
    for pos, idx in enumerate(valid_indices):
        full_to_valid[idx] = pos
    model.eval()
    total_cost = 0
    total_seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            valid_logits = logits.index_select(1, valid_indices_t.to(logits.device))
            order = torch.argsort(valid_logits, dim=1, descending=True)
            positions = torch.empty_like(order)
            ranks = torch.arange(order.size(1), device=order.device).unsqueeze(0).expand_as(order)
            positions.scatter_(1, order, ranks)
            target_cols = full_to_valid[yb].to(order.device)
            target_pos = positions.gather(1, target_cols.unsqueeze(1)).squeeze(1)
            total_cost += int(target_pos.sum().item())
            total_seen += xb.size(0)
    avg = total_cost / total_seen if total_seen else 0.0
    return total_cost, total_seen, avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="input.txt")
    parser.add_argument("--context-len", type=int, default=3)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with open(args.text, encoding="utf-8") as f:
        raw = f.read()
    stream = normalize(raw)
    alphabet, char_to_idx, idx_to_char = build_vocab(stream)
    if len(alphabet) <= 2:
        raise ValueError("The input text does not contain enough characters after normalization.")
    split = max(int(len(stream) * (1 - args.val_fraction)), args.context_len + 1)
    split = min(split, len(stream) - 1)
    train_stream = stream[:split]
    val_stream = stream[split - args.context_len:]
    train_x, train_y = make_examples(train_stream, args.context_len, char_to_idx)
    val_x, val_y = make_examples(val_stream, args.context_len, char_to_idx)
    global_order = build_global_freq(train_stream, alphabet)
    counts = build_ngram_counts([SENTINEL] * args.context_len + train_stream, args.context_len)
    static_fn = static_reorder(global_order)
    oracle_fn = make_oracle(counts, global_order, args.context_len)
    static_cost, static_chars, static_avg = simulate(val_stream, alphabet, static_fn, args.context_len)
    oracle_cost, oracle_chars, oracle_avg = simulate(val_stream, alphabet, oracle_fn, args.context_len)
    model = ContextEmbeddingModel(len(alphabet), args.dim, args.context_len, char_to_idx[SENTINEL])
    model = train_model(model, train_x, train_y, val_x, val_y, args.epochs, args.batch_size, args.lr, args.seed)
    learned_loss = evaluate_loss(model, val_x, val_y, args.batch_size)
    learned_cost, learned_chars, learned_avg = evaluate_rotations(model, val_x, val_y, idx_to_char, char_to_idx, args.batch_size)
    print()
    print(f"Alphabet size: {len(alphabet) - 1}")
    print(f"Context length: {args.context_len}")
    print(f"Embedding dimension: {args.dim}")
    print(f"Train examples: {len(train_x)}")
    print(f"Validation examples: {len(val_x)}")
    print(f"Static baseline: {static_avg:.4f} rotations/char")
    print(f"Oracle 3-gram:    {oracle_avg:.4f} rotations/char")
    print(f"Learned model:    {learned_avg:.4f} rotations/char")
    print(f"Validation CE:    {learned_loss:.4f}")
    print(f"Approx parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Approx float32 flash: {model.parameter_bytes()} bytes")
    sample = val_stream[: args.context_len + 12]
    if len(sample) >= args.context_len:
        context = sample[:args.context_len]
        ordering = model.predict_ordering([char_to_idx[c] for c in context], idx_to_char)
        print()
        print(f"Sample context: {''.join(context)}")
        print(f"Top 10 predictions: {''.join(ordering[:10])}")

if __name__ == "__main__":
    main()

