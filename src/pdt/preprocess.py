from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve.parents[2]

DATA_DIR = REPO_ROOT / "data"
TARGET_LIST_PATH = DATA_DIR / 'deck_hard_list.txt' 
SAY_VOCAB_PATH = DATA_DIR / 'say_thing_vocabulary.txt'

PREP_TARGET_PATH = DATA_DIR / 'prep_targets.npz'

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EVAL_FRACTION = 0.2
SPLIT_SEED = 1337
