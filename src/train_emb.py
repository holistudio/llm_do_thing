import random

import torch
import tiktoken

from llm.gpt2.gpt import GPT

LLM_emb = ...

LLM_speaker = GPT()
speaker_tokenizer = tiktoken.get_encoding('gpt2')

speaker_loss_function = ...
speaker_optimizer = ...
speaker_scheduler = ...

TARGET_WORD_LIST = word_list

SAY_THING_VOCAB = say_thing_vocab

target_word_embs = torch.tensor([LLM_emb(word) for word in TARGET_WORD_LIST])

# choose random word in target word list
# get target_word_emb

# pass to tokenizer

# pass tokens to LLM_speaker
# get logits, log_probs

# use logits/probs to sample from SAY_THING_VOCAB
# get next word
# get corresponding log_prob

# get word_seq


# given_word_seq = word_seq[1:]

# pass given_word_seq to LLM_emb
# get word_seq_emb

# perform cosine similarity b/w word_seq_emb and all target_word_embs
# get best guess word embedding

# reward = cosine_similarity(target_word_emb, guess_word_emb)

# loss = -reward * log_prob

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# scheduler.step()
