import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    words = word_tokenize(example["text"])
    transformed_words = []

    # Dictionary for formal replacements
    informal_to_formal = {
        "awesome": "impressive",
        "cool": "excellent",
        "best": "greatest",
        "movie": "film",
        "amazing": "incredible",
    }

    for word in words:
        # Randomly decide if we should transform this word
        if random.random() < 0.3:
            # 50% chance of synonym replacement
            if random.random() < 0.5:
                synsets = wordnet.synsets(word)
                if synsets:
                    synonyms = synsets[0].lemma_names()
                    if synonyms:
                        synonym = random.choice(synonyms)
                        transformed_words.append(synonym)
                    else:
                        transformed_words.append(word)
                else:
                    transformed_words.append(word)

            # 50% chance of typo introduction
            else:
                if len(word) > 1:
                    typo_word = list(word)
                    random_idx = random.randint(0, len(word) - 1)
                    typo_replacements = {'a': 's', 'e': 'r', 'i': 'o', 'o': 'i', 'u': 'y'}
                    typo_word[random_idx] = typo_replacements.get(word[random_idx], word[random_idx])
                    transformed_words.append("".join(typo_word))
                else:
                    transformed_words.append(word)
        
        # Formalization based on predefined dictionary
        elif word.lower() in informal_to_formal:
            transformed_words.append(informal_to_formal[word.lower()])
        
        # If no transformation, add the word as it is
        else:
            transformed_words.append(word)

    # Detokenize the list of words into a single string
    example["text"] = TreebankWordDetokenizer().detokenize(transformed_words)
    return example

