import math
import random
import torch
import torch.nn as nn
from collections import defaultdict
from utils import normalize, build_vocab, build_global_freq

text_path = "input.txt"
context_len = 3
dim = 8
epochs = 80
batch_size = 128
lr = 0.007
val_fraction = 0.1
seed = 1
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    padding = [" "] * context_len
    padded = padding + stream
    candidate_set = set(alphabet)
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
    padded = [" "] * context_len + stream
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

def split_examples(x, y, val_fraction, seed):
    n = x.size(0)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_idx = torch.tensor(indices[:n_val], dtype=torch.long)
    train_idx = torch.tensor(indices[n_val:], dtype=torch.long)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

class ContextFactorModel(nn.Module):
    def __init__(self, vocab_size, dim, context_len):
        super().__init__()
        self.context_len = context_len
        self.slot_embs = nn.ModuleList([nn.Embedding(vocab_size, dim) for _ in range(context_len)])
        self.char_emb = nn.Embedding(vocab_size, dim)
        self.char_bias = nn.Parameter(torch.zeros(vocab_size))
    def forward(self, context_idxs):
        ctx_vec = 0
        for i in range(self.context_len):
            ctx_vec = ctx_vec + self.slot_embs[i](context_idxs[:, i])
        return ctx_vec @ self.char_emb.weight.t() + self.char_bias
    def predict_ordering(self, context_idxs, idx_to_char):
        device = next(self.parameters()).device
        x = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x).squeeze(0)
            order = torch.argsort(logits, descending=True).tolist()
        return [idx_to_char[i] for i in order]
    def parameter_bytes(self):
        return sum(p.numel() for p in self.parameters()) * 4
    def runtime_bytes(self):
        return sum(p.numel() for p in self.parameters()) * 4

def rank_loss(logits, targets):
    target_scores = logits.gather(1, targets.unsqueeze(1))
    soft_rank = torch.sigmoid(logits - target_scores).sum(dim=1) - 0.5
    return soft_rank.mean()

def evaluate_rank_loss(model, x, y, batch_size):
    model.eval()
    total_loss = 0.0
    n = x.size(0)
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            loss = rank_loss(logits, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / max(n, 1)

def train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_state = None
    best_val = math.inf
    n = train_x.size(0)
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=rng)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = rank_loss(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * xb.size(0)
        scheduler.step()
        train_loss = total_loss / n
        val_loss = evaluate_rank_loss(model, val_x, val_y, batch_size)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:02d}  train {train_loss:.4f}  val {val_loss:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_rotations(model, x, y, batch_size):
    device = next(model.parameters()).device
    model.eval()
    total_cost = 0
    n = x.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            order = torch.argsort(logits, dim=1, descending=True)
            positions = torch.empty_like(order)
            ranks = torch.arange(order.size(1), device=device).unsqueeze(0).expand_as(order)
            positions.scatter_(1, order, ranks)
            target_pos = positions.gather(1, yb.unsqueeze(1)).squeeze(1)
            total_cost += int(target_pos.sum().item())
    avg = total_cost / n if n else 0.0
    return total_cost, n, avg

def main():
    with open(text_path, encoding="utf-8") as f:
        raw = f.read()
    stream = normalize(raw)
    alphabet, char_to_idx, idx_to_char = build_vocab(stream)
    if len(alphabet) <= 1:
        raise ValueError("The input text does not contain enough characters after normalization.")
    all_x, all_y = make_examples(stream, context_len, char_to_idx)
    train_x, train_y, val_x, val_y = split_examples(all_x, all_y, val_fraction, seed)
    global_order = build_global_freq(stream, alphabet)
    counts = build_ngram_counts([" "] * context_len + stream, context_len)
    static_fn = static_reorder(global_order)
    oracle_fn = make_oracle(counts, global_order, context_len)
    static_cost, static_chars, static_avg = simulate(stream, alphabet, static_fn, context_len)
    oracle_cost, oracle_chars, oracle_avg = simulate(stream, alphabet, oracle_fn, context_len)
    print(f"Model device:        {device}")
    print(f"Alphabet:            {''.join(global_order)}")
    print(f"Alphabet size:       {len(alphabet)}")
    print(f"Context length:      {context_len}")
    print(f"Dim:                 {dim}")
    print(f"Train examples:      {len(train_x)}")
    print(f"Validation examples: {len(val_x)}")
    print(f"Static baseline:     {static_avg:.4f} rotations/char")
    print(f"Oracle {context_len}-gram:       {oracle_avg:.4f} rotations/char")
    model = ContextFactorModel(len(alphabet), dim, context_len).to(device)
    model = train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed)
    cost, chars, avg = evaluate_rotations(model, val_x, val_y, batch_size)
    print(f"\nVal rotations/char:  {avg:.4f}  (vs static {static_avg - avg:+.4f}, vs oracle {oracle_avg - avg:+.4f})")
    print(f"Flash bytes:         {model.parameter_bytes()}")
    print(f"Runtime bytes:       {model.runtime_bytes()}")
    sample_ctx = [idx_to_char[i.item()] for i in val_x[0]]
    sample_idxs = val_x[0].tolist()
    ordering = model.predict_ordering(sample_idxs, idx_to_char)
    print(f"\nSample context: {''.join(sample_ctx)}")
    print(f"Predictions: {''.join(ordering)}")

if __name__ == "__main__":
    main()
