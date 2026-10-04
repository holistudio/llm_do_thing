from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"

SAY_VOCAB_PATH = DATA_DIR / 'say_thing_vocabulary.txt'

def load_words(path):
    words = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()]
    words = [w for w in words if w]
    assert len(words) == len(set(words)), f"{path.name} has duplicate words"
    return words

vocab = load_words(SAY_VOCAB_PATH)
print(f'say thing vocab has {len(vocab)} words\n') # just a sanity check

TARGET_LIST_PATH = DATA_DIR / 'deck_hard_list.txt' 
PREP_TARGET_PATH = DATA_DIR / 'prep_targets.npz'

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EVAL_FRACTION = 0.2
SPLIT_SEED = 1337

def embed_words(words, model_name=EMBED_MODEL, device="cpu"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        words,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings.astype(np.float32)

def similarity_matrix(embeddings):
    S = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S.astype(np.float32)

def off_diagonal_means(S):
    n = S.shape[0]
    row_sums = S.sum(axis=1) - np.diagonal(S)
    return (row_sums / (n-1)).astype(np.float32)

def train_eval_split(n, eval_fraction=EVAL_FRACTION, seed=SPLIT_SEED):
    order = np.random.default_rng(seed).permutation(n)
    n_eval = round(n * eval_fraction)
    train_idx = order[n_eval:]
    eval_idx = order[:n_eval]
    return train_idx, eval_idx

def report(words, S, mu, k=8):
    # report reward's dynamic range
    n = len(words)
    print(f"mu:  min {mu.min():.3f}  mean {mu.mean():.3f}  max {mu.max():.3f}")

    # max off-diagonal near 1.0 means a near-duplicate pair the speaker can never disambiguate
    off = S[~np.eye(n, dtype=bool)]
    print(f"off-diagonal cosine:  min {off.min():.3f}  max {off.max():.3f}")

    # closest pairs reveal where "embedding artifacts" risk near-miss
    pairs = np.triu(S, k=1)
    top = np.argsort(pairs, axis=None)[::-1][:k]
    for i, j in zip(*np.unravel_index(top, pairs.shape)):
        print(f"  {S[i, j]:.3f}  {words[i]:>14} ~ {words[j]}")

    # high mean means a lazy speaker scores well without ever being right
    nearest = np.where(np.eye(n, dtype=bool), -np.inf, S).max(axis=1)
    s_hat = (nearest - mu) / (1.0 - mu)
    print(f"nearest-neighbour s_hat:  mean {s_hat.mean():.3f}  max {s_hat.max():.3f}")


def save(words, embeddings, S, mu, train_idx, eval_idx, path=PREP_TARGET_PATH):
    np.savez(
        path,
        words=np.array(words),
        embeddings=embeddings,
        similarities=S,
        means=mu,
        train_idx=train_idx.astype(np.int32),
        eval_idx=eval_idx.astype(np.int32),
        model_name=EMBED_MODEL,
        eval_fraction=EVAL_FRACTION,
        split_seed=SPLIT_SEED,
    )
    return path


def load_prep(path=PREP_TARGET_PATH):
    with np.load(path, allow_pickle=False) as f:
        prep = {k: f[k] for k in f.files}
    prep["words"] = [str(w) for w in prep["words"]]
    assert prep["words"] == load_words(TARGET_LIST_PATH), "prep is stale, rerun step 0"
    return prep

if __name__ == '__main__':
    targets = load_words(TARGET_LIST_PATH)
    n = len(targets)
    emb_targets = embed_words(targets)
    S = similarity_matrix(emb_targets)
    mu = off_diagonal_means(S)
    train_idx, eval_idx = train_eval_split(n)
    report(targets, S, mu)
    save(targets, emb_targets, S, mu, train_idx, eval_idx)

