# episode sequence

# sample random word from training split of TARGET_WORD_LIST

# give word to LLM_speaker with system prompt, SAY_THING_VOCAB
# user prompt with random TARGET_WORD, say_tool

# validate LLM_speaker's output consists only of SAY_THING_VOCAB
# increment n_retries if not

# if validation passes, give LLM_speaker output to LLM_guesser
# system prompt and TARGET_WORD_LIST are given to LLM_guesser
# LLM_guesser outputs GUESS_WORD

# compute reward based on n_retries, GUESS_WORD

# after batch of episodes records full trace
# update weights of LLM_speaker based on reward signals

# after EVAL_INTERVAL training episodes
# evaluate LLM_speaker on evaluation split of TARGET_WORD_LIST