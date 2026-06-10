import math
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OMIT = frozenset("q")
SENTINEL = "^"
text_path = "input.txt"
context_len = 3
dim = 4
epochs = 80
batch_size = 128
lr = 0.007
val_fraction = 0.1
seed = 1
random.seed(seed)
torch.manual_seed(seed)

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

def split_examples(x, y, val_fraction, seed):
    n = x.size(0)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_idx = torch.tensor(indices[:n_val], dtype=torch.long)
    train_idx = torch.tensor(indices[n_val:], dtype=torch.long)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

def expected_rank_loss(logits, targets, sentinel_idx):
    batch = logits.size(0)
    vocab = logits.size(1)
    mask = torch.ones(vocab, dtype=torch.bool, device=logits.device)
    mask[sentinel_idx] = False
    valid_logits = logits[:, mask]
    valid_size = valid_logits.size(1)
    full_to_valid = torch.full((vocab,), -1, dtype=torch.long, device=logits.device)
    valid_indices = torch.where(mask)[0]
    full_to_valid[valid_indices] = torch.arange(valid_size, device=logits.device)
    target_valid = full_to_valid[targets]
    target_scores = valid_logits[torch.arange(batch, device=logits.device), target_valid].unsqueeze(1)
    margins = valid_logits - target_scores
    self_mask = torch.ones_like(margins, dtype=torch.bool)
    self_mask[torch.arange(batch, device=logits.device), target_valid] = False
    loss = (torch.sigmoid(margins) * self_mask).sum(dim=1)
    return loss.mean()

class CharOrderModel(nn.Module):
    def __init__(self, vocab_size, dim, context_len, sentinel_idx):
        super().__init__()
        self.context_len = context_len
        self.sentinel_idx = sentinel_idx
        self.char_emb = nn.Embedding(vocab_size, dim)
        self.output_emb = nn.Embedding(vocab_size, dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.register_buffer('pos_coeffs', torch.linspace(0.5, 1.5, context_len).view(1, context_len, 1))
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        char_vecs = self.char_emb(x)
        weighted = char_vecs * self.pos_coeffs
        state = weighted.sum(dim=1)
        state_norm = state / (state.norm(dim=1, keepdim=True) + 1e-8)
        out_norm = self.output_emb.weight / (self.output_emb.weight.norm(dim=1, keepdim=True) + 1e-8)
        logits = (state_norm @ out_norm.t()) * 10.0 + self.output_bias
        logits[:, self.sentinel_idx] = -1e9
        return logits
    def predict_ordering(self, context_ids, idx_to_char):
        device = next(self.parameters()).device
        x = torch.tensor(context_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.forward(x).squeeze(0)
            order = torch.argsort(logits, descending=True).tolist()
        return [idx_to_char[i] for i in order if i != self.sentinel_idx]
    def parameter_bytes(self):
        return sum(p.numel() for p in self.parameters()) * 4
    def runtime_bytes(self):
        state_bytes = (self.char_emb.embedding_dim) * 4
        return state_bytes

def train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_state = None
    best_val = math.inf
    n = train_x.size(0)
    rng = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=rng)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = train_x[idx], train_y[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = expected_rank_loss(logits, yb, model.sentinel_idx)
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

def evaluate_rank_loss(model, x, y, batch_size):
    model.eval()
    total_loss = 0.0
    n = x.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
            logits = model(xb)
            loss = expected_rank_loss(logits, yb, model.sentinel_idx)
            total_loss += loss.item() * xb.size(0)
    return total_loss / max(n, 1)

def evaluate_rotations(model, x, y, idx_to_char, char_to_idx, batch_size):
    valid_indices = [i for i in range(len(idx_to_char)) if i != char_to_idx[SENTINEL]]
    valid_indices_t = torch.tensor(valid_indices, dtype=torch.long)
    full_to_valid = torch.full((len(idx_to_char),), -1, dtype=torch.long)
    for pos, idx in enumerate(valid_indices):
        full_to_valid[idx] = pos
    model.eval()
    total_cost = 0
    n = x.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
            logits = model(xb)
            valid_logits = logits.index_select(1, valid_indices_t.to(logits.device))
            order = torch.argsort(valid_logits, dim=1, descending=True)
            positions = torch.empty_like(order)
            ranks = torch.arange(order.size(1), device=order.device).unsqueeze(0).expand_as(order)
            positions.scatter_(1, order, ranks)
            target_cols = full_to_valid[yb].to(order.device)
            target_pos = positions.gather(1, target_cols.unsqueeze(1)).squeeze(1)
            total_cost += int(target_pos.sum().item())
    avg = total_cost / n if n else 0.0
    return total_cost, n, avg

def main():
    with open(text_path, encoding="utf-8") as f:
        raw = f.read()
    stream = normalize(raw)
    alphabet, char_to_idx, idx_to_char = build_vocab(stream)
    if len(alphabet) <= 2:
        raise ValueError("The input text does not contain enough characters after normalization.")
    all_x, all_y = make_examples(stream, context_len, char_to_idx)
    train_x, train_y, val_x, val_y = split_examples(all_x, all_y, val_fraction, seed)
    global_order = build_global_freq(stream, alphabet)
    counts = build_ngram_counts([SENTINEL] * context_len + stream, context_len)
    static_fn = static_reorder(global_order)
    oracle_fn = make_oracle(counts, global_order, context_len)
    static_cost, static_chars, static_avg = simulate(stream, alphabet, static_fn, context_len)
    oracle_cost, oracle_chars, oracle_avg = simulate(stream, alphabet, oracle_fn, context_len)
    print(f"Alphabet: {''.join(global_order)}")
    print(f"Alphabet size:       {len(alphabet) - 1}")
    print(f"Context length:      {context_len}")
    print(f"Dim:                 {dim}")
    print(f"Train examples:      {len(train_x)}")
    print(f"Validation examples: {len(val_x)}")
    print(f"Static baseline:     {static_avg:.4f} rotations/char")
    print(f"Oracle {context_len}-gram:       {oracle_avg:.4f} rotations/char")
    model = CharOrderModel(len(alphabet), dim, context_len, char_to_idx[SENTINEL])
    model = train_model(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, seed)
    cost, chars, avg = evaluate_rotations(model, val_x, val_y, idx_to_char, char_to_idx, batch_size)
    print(f"\nVal rotations/char:  {avg:.4f}  (vs static {static_avg - avg:+.4f}, vs oracle {oracle_avg - avg:+.4f})")
    print(f"Flash bytes:         {model.parameter_bytes()}")
    print(f"Runtime bytes:       {model.runtime_bytes()}")
    sample_ctx = [idx_to_char[i.item()] for i in val_x[0]]
    ordering = model.predict_ordering(val_x[0].tolist(), idx_to_char)
    print(f"\nSample context: {''.join(sample_ctx)}")
    print(f"Predictions: {''.join(ordering)}")

if __name__ == "__main__":
    main()
