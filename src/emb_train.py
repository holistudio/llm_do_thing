import copy

import numpy as np
import torch
import torch.nn.functional as F

import tiktoken
from sentence_transformers import SentenceTransformer, util

from llm.gpt2.gpt import GPT_SayThing


def main(word_list, say_thing_vocab, train_epochs=1000, num_rounds=10):
    torch.manual_seed(1337)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    TARGET_WORD_LIST = word_list
    SAY_THING_VOCAB = say_thing_vocab

    # initialize speaker
    LLM_speaker = GPT_SayThing().to(device) # output shape (bs, T, 33)
    speaker_tokenizer = tiktoken.get_encoding('gpt2')
    speaker_optimizer = torch.optim.AdamW(LLM_speaker.parameters(), lr=1e-6, weight_decay=0.1) #lr=3e-4 TOO HIGH

    # initialize embedding model to serve as Guesser
    # and reward function computations
    print("\nLoading embedding model...")
    LLM_emb = SentenceTransformer("all-MiniLM-L6-v2")
    target_word_embs = torch.tensor(np.array([LLM_emb.encode(word) for word in TARGET_WORD_LIST]), device=device)

    for ep in range(train_epochs):
        # choose random word in target word list
        # get embedding vector
        target_ix = torch.randint(0,len(TARGET_WORD_LIST),(1,)).item()
        target_word = TARGET_WORD_LIST[target_ix]
        target_word_emb = target_word_embs[target_ix]

        # initialize input string for LLM_speaker
        speak_str = target_word

        # initialize input for LLM_emb
        given_word_seq =  ""
        guesser_word_bag = copy.deepcopy(TARGET_WORD_LIST)
        guesser_emb_bag = copy.deepcopy(target_word_embs)

        for _ in range(num_rounds):
            # pass input string to tokenizer
            speak_token_ids = speaker_tokenizer.encode(speak_str)
            speak_token_ids = torch.tensor(speak_token_ids, dtype=torch.long, device=device).unsqueeze(0)

            # pass tokens to LLM_speaker
            speak_logits = LLM_speaker(speak_token_ids)
            # get logits, log_probs
            speak_logits = speak_logits[:, -1, :]
            log_probs = F.log_softmax(speak_logits, dim=-1)

            # use logits/probs to sample from SAY_THING_VOCAB
            probs = torch.exp(log_probs)
            sample_ix = torch.multinomial(probs, num_samples=1)

            # get next word chosen by LLM_speaker
            next_word = SAY_THING_VOCAB[sample_ix]
            speak_str += " " + next_word
            # get corresponding log_prob
            log_prob = log_probs[0, sample_ix]

            # get word seq to give to the guesser/LLM_emb
            word_seq = speak_str.split(" ")
            given_word_seq = " ".join(word_seq[1:])
            word_seq_emb = LLM_emb.encode(given_word_seq, convert_to_tensor=True).to(device=device)

            # perform cosine similarity b/w word_seq_emb and all guesser_emb_bag
            scores = util.cos_sim(word_seq_emb, guesser_emb_bag)
            # get best guess word embedding based on cosine similarity
            guess_ix = torch.argmax(scores)
            guess_word_emb = guesser_emb_bag[guess_ix]
            guess_word = guesser_word_bag[guess_ix]

            # compute reward feedback based on cosine similarity
            # between target word and guess
            reward = util.cos_sim(target_word_emb, guess_word_emb)

            # compute loss for LLM_speaker
            speaker_loss = -reward * log_prob

            speaker_optimizer.zero_grad()
            speaker_loss.backward()
            speaker_optimizer.step()

            if (ep+1) % 100 == 0:
                print(f"episode {ep}: {target_word}")
                print(f"{given_word_seq}, prob={torch.exp(log_prob).item():.3f}, log_prob={log_prob.item()}")
                print(f"{guess_word}, similarity score={reward.item():.2f}")
                print(f"loss={speaker_loss.item()}")
                print()

            # game ends early if LLM_speaker outputs the same Say-Thing
            # vocabulary word 3 times
            if len(word_seq) >= 4 and (word_seq[-1] == word_seq[-2]) and (word_seq[-1] == word_seq[-3]):
                # print('3 strikes')
                break

            # game ends if guesser guesses the word correctly
            if target_word == guess_word:
                break
            else:
                # remove guessed word from possible future guesses in this episode
                guesser_word_bag.pop(guess_ix)
                guesser_emb_bag = torch.cat((guesser_emb_bag[:guess_ix], guesser_emb_bag[guess_ix+1:]), dim=0)


if __name__ == "__main__":
    with open("website_word_list.txt", "r") as f:
        word_list = f.read().split('\n')

    with open("say_thing_vocabulary.txt", "r") as f:
        say_thing_vocab = f.read().split('\n')

    main(word_list, say_thing_vocab)