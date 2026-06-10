from collections import defaultdict, Counter
from math import log2

OMIT = frozenset('qx')

def build_alphabet(text):
    seen = set(text)
    seen.discard('')
    return sorted(seen)

def normalize(raw, omit=OMIT):
    out = []
    for ch in raw:
        if ch in (' ', '\n'):
            out.append(' ')
        elif ch.isalpha():
            low = ch.lower()
            if low not in omit:
                out.append(low)
    return out

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
        if ch in alphabet:
            freq[ch] += 1
    total = sum(freq.values()) or 1
    return sorted(alphabet, key=lambda c: -freq[c] / total)

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

def simulate(stream, alphabet, reorder_fn, max_order):
    total_cost = 0
    total_chars = 0
    for i in range(len(stream)):
        ch = stream[i]
        if ch not in alphabet:
            continue
        start = max(0, i - max_order)
        context = stream[start:i]
        ordering = reorder_fn(context)
        try:
            pos = ordering.index(ch)
        except ValueError:
            pos = len(ordering)
        total_cost += pos
        total_chars += 1
    avg = total_cost / total_chars if total_chars else 0
    return total_cost, total_chars, avg

def static_reorder(global_order):
    def reorder(context):
        return global_order
    return reorder

def build_oracle_table(stream, oracle_fn, order):
    table = {}
    for i in range(len(stream)):
        start = max(0, i - order)
        ctx = tuple(stream[start:i])
        if ctx not in table:
            table[ctx] = tuple(oracle_fn(ctx))
    return table

def kendall_distance(a, b):
    pos_b = {c: i for i, c in enumerate(b)}
    inv = 0
    n = len(a)
    for i in range(n):
        ai = a[i]
        for j in range(i + 1, n):
            aj = a[j]
            if pos_b[ai] > pos_b[aj]:
                inv += 1
    return inv

def entropy_from_counter(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counter.values():
        p = count / total
        h -= p * log2(p)
    return h

def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    idx = int((len(values) - 1) * p)
    return values[idx]

def analyze_prediction(stream, oracle_fn, order):
    ranks = []
    top1 = 0
    top3 = 0
    top5 = 0
    top8 = 0
    total = 0
    for i in range(len(stream)):
        ch = stream[i]
        start = max(0, i - order)
        context = stream[start:i]
        ordering = oracle_fn(context)
        try:
            rank = ordering.index(ch)
        except ValueError:
            rank = len(ordering)
        ranks.append(rank)
        top1 += rank < 1
        top3 += rank < 3
        top5 += rank < 5
        top8 += rank < 8
        total += 1
    mean_rank = sum(ranks) / len(ranks) if ranks else 0
    print("True next-letter rank:")
    print(f"  mean   : {mean_rank:.3f}")
    print(f"  median : {percentile(ranks, 0.5)}")
    print(f"  p95    : {percentile(ranks, 0.95)}")
    print("Top-k accuracy:")
    print(f"  top-1 : {100.0 * top1 / total:.2f}%")
    print(f"  top-3 : {100.0 * top3 / total:.2f}%")
    print(f"  top-5 : {100.0 * top5 / total:.2f}%")
    print(f"  top-8 : {100.0 * top8 / total:.2f}%")

def analyze_oracle(stream, oracle_fn, global_order, order):
    table = build_oracle_table(stream, oracle_fn, order)
    print(f"=== Oracle analysis (order={order}) ===")
    print(f"Contexts observed: {len(table)}")
    unique_orderings = Counter(table.values())
    print(f"Unique orderings: {len(unique_orderings)}")
    most_common_ordering, count = unique_orderings.most_common(1)[0]
    pct = 100.0 * count / len(table)
    print(f"Most common ordering: {pct:.2f}% of contexts")
    top1 = Counter()
    top3 = Counter()
    top5 = Counter()
    for ordering in table.values():
        top1[ordering[:1]] += 1
        top3[ordering[:3]] += 1
        top5[ordering[:5]] += 1
    print(f"Distinct top-1 prefixes: {len(top1)}")
    print(f"Distinct top-3 prefixes: {len(top3)}")
    print(f"Distinct top-5 prefixes: {len(top5)}")
    global_pos = {c: i for i, c in enumerate(global_order)}
    displacement_sum = 0
    displacement_count = 0
    for ordering in table.values():
        for pos, ch in enumerate(ordering):
            if ch not in global_pos:
                continue
            displacement_sum += abs(pos - global_pos[ch])
            displacement_count += 1
    avg_displacement = displacement_sum / displacement_count if displacement_count else 0
    print(f"Average displacement from global ordering: {avg_displacement:.3f}")
    alphabet_size = len(global_order)
    print("Agreement with global ordering:")
    for k in [1, 3, 5, 10]:
        matches = 0
        total = 0
        for ordering in table.values():
            for pos in range(min(k, alphabet_size, len(ordering))):
                if ordering[pos] == global_order[pos]:
                    matches += 1
                total += 1
        agreement = 100.0 * matches / total if total else 0
        print(f"  top-{k:<2}: {agreement:.2f}%")
    print("Top-N divergence from global ordering:")
    for k in [3, 5, 8, 10]:
        diff_sum = 0
        hist = Counter()
        for ordering in table.values():
            limit = min(k, len(ordering), len(global_order))
            diff = sum(1 for i in range(limit) if ordering[i] != global_order[i])
            diff_sum += diff
            hist[diff] += 1
        avg_diff = diff_sum / len(table)
        print(f"  top-{k}: avg differing positions = {avg_diff:.3f}")
        for diff in sorted(hist):
            pct = 100.0 * hist[diff] / len(table)
            print(f"    {diff} differences: {pct:.2f}%")
    print("Top-N membership changes:")
    for k in [3, 5, 8, 10]:
        changes = []
        global_top = set(global_order[:k])
        for ordering in table.values():
            oracle_top = set(ordering[:k])
            changes.append(len(oracle_top.symmetric_difference(global_top)) // 2)
        avg_changes = sum(changes) / len(changes)
        print(f"  top-{k}: avg letters replaced = {avg_changes:.3f}")
    distances = []
    for ordering in table.values():
        filtered = [ch for ch in ordering if ch in global_pos]
        if len(filtered) == alphabet_size:
            distances.append(kendall_distance(filtered, global_order))
    mean_distance = sum(distances) / len(distances) if distances else 0
    max_distance = alphabet_size * (alphabet_size - 1) // 2
    print(f"Average Kendall distance: {mean_distance:.2f}")
    print(f"Normalized Kendall distance: {100.0 * mean_distance / max_distance:.2f}%")
    print(f"Average adjacent swaps from global ordering: {mean_distance:.2f}")
    first_letters = Counter()
    for ordering in table.values():
        if ordering:
            first_letters[ordering[0]] += 1
    print("Most common first-place letters:")
    for ch, count in first_letters.most_common(10):
        pct = 100.0 * count / len(table)
        print(f"  {repr(ch):>3} : {pct:6.2f}%")
    top5_letters = Counter()
    for ordering in table.values():
        for ch in ordering[:5]:
            top5_letters[ch] += 1
    total_slots = len(table) * 5
    print("Most common top-5 letters:")
    for ch, count in top5_letters.most_common(10):
        pct = 100.0 * count / total_slots
        print(f"  {repr(ch):>3} : {pct:6.2f}%")
    position_counts = defaultdict(Counter)
    max_pos = min(10, alphabet_size)
    for ordering in table.values():
        for pos in range(max_pos):
            position_counts[pos][ordering[pos]] += 1
    print("Positional entropy:")
    for pos in range(max_pos):
        h = entropy_from_counter(position_counts[pos])
        eff = 2 ** h
        print(f"  pos {pos + 1:2d}: {h:.3f} bits, effective choices {eff:.2f}")

def main():
    with open('input.txt', encoding='utf-8') as f:
        raw = f.read()
    stream = normalize(raw)
    alphabet = sorted(set(stream))
    global_order = build_global_freq(stream, alphabet)
    print(f"Alphabet size: {len(alphabet)}")
    print(f"Stream length: {len(stream)}")
    print(f"Global frequency order: {''.join(global_order)}")
    max_order = 4
    counts = build_ngram_counts(stream, max_order)
    static_fn = static_reorder(global_order)
    _, _, static_avg = simulate(stream, set(alphabet), static_fn, max_order)
    print(f"Static alphabet baseline: {static_avg:.4f} avg rotations/char")
    for order in range(1, max_order + 1):
        oracle_fn = make_oracle(counts, global_order, order)
        _, _, oracle_avg = simulate(stream, set(alphabet), oracle_fn, order)
        reduction = 100 * (1 - oracle_avg / static_avg)
        print(f"Oracle (max order={order}):         {oracle_avg:.4f} avg rotations/char  ({reduction:.1f}% reduction)")
        analyze_prediction(stream, oracle_fn, order)
        analyze_oracle(stream, oracle_fn, global_order, order)

if __name__ == '__main__':
    main()
