# llm_do_thing

This is the word game *Person Do Thing*:

<img src="./docs/card_game.png" width=800px>

Imagine you see the word "fireworks" on a card. This is your *target word* that I have to guess. The only words you are allowed to say are the 33 words on the bigger card shown above, the **Say Thing Vocabulary**.

How would you describe fireworks using only those 33 words? Maybe "hot thing go far up fast after big many hot thing"?

(you can also play [person-do-thing](https://persondothing.com/) online with friends)

## The Idea

One day as I played this game with friends at [The Recurse Center](https://www.recurse.com/), I started to think about how to train LLMs to also do this and more specifically, how an LLM can be finetuned to output a sequence of words from the *Say Thing Vocabulary*. How would the finetuning process work? 

And more importantly, what words would the LLM end up choosing? Would we mere humans also be able to tell what word the LLM is trying to communicate?

## The Current Approach

The current idea is to use some form of reinforcement learning to finetune a pre-trained LLM to output a sequence of words restricted to among the 33 possible words in the *Say Thing Vocabulary*:

<img src="./docs/260604_RL_idea.png">

0. Let's assume the list of possible target words is limited to the 64 possible words from [persondothing.com](https://persondothing.com/) listed in [website_word_list.txt](./src/website_word_list.txt) and that the guesser has access to this list.
1. After receiving one of the possible target words as initial input, `LLM_speaker` spits out a sequence of words among the limited list in *Say Thing Vocabulary*.
2. Each time `LLM_speaker` adds a word to the word sequence, an `LLM_embedding` model can convert the word sequence into an embedding vector and compare that to all possible target words' embeddings via cosine similarity. The guess word is the one with the highest cosine similarity with the `LLM_speaker`'s word sequence.
3. A reward function provides feedback signal back to the `LLM_speaker` based on the cosine similarity between the target and guess word.
4. This can continue for let's say up to 10 rounds, or until the guess word is the same as the target word. Then the game resets with a new random target word.

That's it for now, this is still very much work in progress. More specifics can be found in the code and [DEVLOG](./docs/DEVLOG.md).


## Install and Run

To try it out yourself, activate a virtual environment (`uv` or `conda`) and install requirements

```bash
(uv) pip install -r requirements.txt
```

Then run the training to see how it's learning over episodes. 

```bash
cd src
python emb_train.py
```

Default training settings:
- 1000 training episodes
- 10 rounds: `LLM_speaker` can output up to 10 words from *Say Thing Vocabulary*.
- "3 strikes" for repeated words: If `LLM_speaker` says the same word three times, the episode ends early (because why would anyone say the same word more than two times if it's not helping the guesser?)
- `LLM_speaker` currently uses the GPT-2 architecture with 124M pre-trained weights from OpenAI (run `python gpt_download.py` within its local directory). For fine tuning, the default is to keep all parameters frozen except the last transformer block/layer and output layer.

Example output:

```
episode 699: whisper # target word for the training episode
use make, prob=0.353, log_prob=-1.041 # for the last word predicted by LLM_speaker
kitchen, similarity score=0.29 # guesser's word and cosine similarity
loss=0.301 # loss feedback to LLM_speaker

episode 699: whisper
use make use, prob=0.625, log_prob=-0.470
harvest, similarity score=0.20
loss=0.092

episode 699: whisper
use make use fast, prob=0.022, log_prob=-3.826
bakery, similarity score=0.15
loss=0.579

episode 699: whisper
use make use fast use, prob=0.275, log_prob=-1.291
whisper, similarity score=1.00
loss=1.291
```

Another hilarious result:

```
episode 399: elephant
real, prob=0.117, log_prob=-2.144
birthday, similarity score=0.24
loss=0.509

episode 399: elephant
real use, prob=0.659, log_prob=-0.417
library, similarity score=0.28
loss=0.117

...

episode 399: elephant
real use thing use use fast thing thing use, prob=0.476, log_prob=-0.742
elephant, similarity score=1.00
loss=0.742
```

Just imagine you had a board in front of you with a list of the possible target words and then some person just said to you "real use thing use use fast thing thing use"...would you be able to know they meant to say "elephant"?

## References

Others at The Recurse Center has [also approached this word game](https://www.scd31.com/posts/person-do-thing-llm) with LLMs as well. I would consider their approach more "agentic AI" than LLM finetuning but still interesting!

This is what excites me the most about this project. There's an entire spectrum of neural network to agentic AI workflow-based solutions to consider.