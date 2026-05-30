import random

import torch

from llm.gpt2.gpt import GPT

def word_sampling(logits, word_complete, tokenizer, word_set=None):
    return word_part, word_complete

def reward_function(target_word, guess_word, given_word_seq, emb_model=foo):
    return score

def main(train_epochs=10, num_rounds=10):
    TARGET_WORD_LIST = ['desert']

    SAY_THING_VOCAB = []

    speaker = GPT()
    guesser = GPT()

    speaker_tokenizer = ...
    guesser_tokenizer = ...

    speaker_loss_function = ...
    speaker_optimizer = ...
    speaker_scheduler = ...

    # guesser_loss_function = ...
    # guesser_optimizer = ...
    # guesser_scheduler = ...

    for ep in range(train_epochs):

        target_word = TARGET_WORD_LIST[random.randint(0,len(TARGET_WORD_LIST)-1)]

        speak_str = target_word

        while len(given_word_seq) < num_rounds:

            speak_word_complete = False
            while not speak_word_complete:
                speak_token_ids = speaker_tokenizer.encode(speak_str)
                speak_token_ids = torch.tensor(speak_token_ids, dtype=torch.long).unsqueeze(0)

                speaker.optimizer.zero_grad()
                speak_logits = speaker(speak_token_ids)
                speak_logits = speak_logits[:, -1, :]

                speak_word_part, speak_word_complete = word_sampling(speak_logits, speak_word_complete, speaker_tokenizer, word_set=SAY_THING_VOCAB)
                # speak_word_part, speak_word_complete = word_sampling(speak_logits, speak_word_complete, word_set=None)
                speak_str += speak_word_part
                
            word_seq = speak_str.split(", ")

            given_word_seq = word_seq[1:]
            guess_str = ", ".join(given_word_seq)
            guess_str += " | "

            guess_word_complete = False
            while not guess_word_complete:
                guess_token_ids = guesser_tokenizer.encode(guess_str)
                guess_token_ids = torch.tensor(guess_token_ids, dtype=torch.long).unsqueeze(0)

                guess_logits = guesser(guess_token_ids)
                guess_logits = guess_logits[:, -1, :]

                guess_word_part, guess_word_complete = word_sampling(guess_logits, guess_word_complete, guesser_tokenizer, word_set=TARGET_WORD_LIST)
                guess_str += guess_word_part

            guess_word = guess_str.split(" | ")[-1]

            reward = reward_function(target_word, guess_word, given_word_seq)

            speaker_loss = speaker_loss_function(reward)
            speaker_loss.backward()

            # guesser_loss = guesser_loss_function(reward)
            # guesser_loss.backward()

            speaker_optimizer.step()
            speaker_scheduler.step()

            # guesser_optimizer.step()
            # guesser_scheduler.step()

if __name__ == "__main__":
    main()