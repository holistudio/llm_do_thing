# llm_do_thing

This is the word game *Person Do Thing*:

<img src="./docs/card_game.png" width=800px>

Imagine you see the word "fireworks" on a card. This is your *target word* that I have to guess. The only words you are allowed to say are the 33 words on the bigger card shown above, the **Say Thing Vocabulary**.

(you can also play [person-do-thing](https://persondothing.com/) online with friends)

## The Idea

One day as I played this game with friends at [The Recurse Center](https://www.recurse.com/), I started to think about how to train LLMs to also do this and more specifically, how an LLM can be finetuned to output a sequence of words from the *Say Thing Vocabulary*. How would the finetuning process work? And more importantly, what words would the LLM end up choosing? Would we mere humans also be able to tell what word the LLM is trying to communicate?

## The Current Approach

The current idea is to use some form of reinforcement learning to finetune a pre-trained LLM to output a sequence of words restricted to among the 33 possible words in the *Say Thing Vocabulary*:

1. After receiving one of the possible target words as initial input, `LLM_speaker` spits out a sequence of words among the limited list of words in *Say Thing Vocabulary*.
2. For each word added to the sequence, an `LLM_guesser` tries to guess the target word.
3. A reward function provides feedback signal back to the `LLM_speaker` based on the cosine similarity between the target word and the latest `LLM_guesser`'s word.

That's it for now, this is still very much work in progress. More specifics can be found in the code and [DEVLOG](./docs/DEVLOG.md).

## References

Others at The Recurse Center has [also approached this word game](https://www.scd31.com/posts/person-do-thing-llm) with LLMs as well. I would consider their approach more "agentic AI" than LLM finetuning but still interesting!

This is what excites me the most about this project. There's an entire spectrum of neural network to agentic AI workflow-based solutions to consider.