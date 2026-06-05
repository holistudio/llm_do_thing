import random

import torch
import torch.nn.functional as F

import tiktoken

from llm.gpt2.gpt import GPT_SayThing

def cosine_similarity(x, vectors):
    return distances

def main(word_list, say_thing_vocab, train_epochs=10, num_rounds=10):
    LLM_emb = ...

    LLM_speaker = GPT_SayThing()
    speaker_tokenizer = tiktoken.get_encoding('gpt2')

    speaker_optimizer = ...
    speaker_scheduler = ...

    TARGET_WORD_LIST = word_list

    SAY_THING_VOCAB = say_thing_vocab

    target_word_embs = torch.tensor([LLM_emb(word) for word in TARGET_WORD_LIST])

    for ep in range(train_epochs):
        # choose random word in target word list
        target_ix = random.randint(0,len(TARGET_WORD_LIST)-1)
        target_word = TARGET_WORD_LIST[target_ix]
        target_word_emb = target_word_embs[target_ix]

        speak_str = target_word

        given_word_seq =  []

        for _ in range(num_rounds):
            # pass to tokenizer
            speak_token_ids = speaker_tokenizer.encode(speak_str)
            speak_token_ids = torch.tensor(speak_token_ids, dtype=torch.long).unsqueeze(0)

            # pass tokens to LLM_speaker
            speak_logits = LLM_speaker(speak_token_ids)
            # get logits, log_probs
            speak_logits = speak_logits[:, -1, :]
            log_probs = F.log_softmax(speak_logits)

            # use logits/probs to sample from SAY_THING_VOCAB
            probs = torch.exp(log_probs)
            sample_ix = torch.multinomial(probs, num_samples=1)

            # get next word
            next_word = SAY_THING_VOCAB[sample_ix]
            speak_str += " " + next_word
            # get corresponding log_prob
            log_prob = log_probs[0, sample_ix]

            # get word_seq
            word_seq = speak_str.split(" ")


            given_word_seq = word_seq[1:]

            # pass given_word_seq to LLM_emb
            word_seq_emb = LLM_emb(given_word_seq)

            # perform cosine similarity b/w word_seq_emb and all target_word_embs
            distances = cosine_similarity(word_seq_emb, target_word_embs)
            # get best guess word embedding
            guess_ix = torch.argmax(distances)
            guess_word_emb = target_word_embs[guess_ix]
            guess_word = TARGET_WORD_LIST[guess_ix]

            reward = cosine_similarity(target_word_emb, guess_word_emb)

            speaker_loss = -reward * log_prob

            speaker_optimizer.zero_grad()
            speaker_loss.backward()
            speaker_optimizer.step()
            speaker_scheduler.step()


if __name__ == "__main__":
    with open("website_word_list.txt", "r") as f:
        word_list = f.read().split('\n')

    with open("say_thing_vocabulary.txt", "r") as f:
        say_thing_vocab = f.read().split('\n')

    main(word_list, say_thing_vocab)