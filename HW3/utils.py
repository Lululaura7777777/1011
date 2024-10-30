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
    new_words = []

    for word in words:
        # Synonym replacement with a 30% chance
        if random.random() < 0.3:
            synonyms = wordnet.synsets(word)
            if synonyms:
                # Pick a random synonym lemma
                synonym = synonyms[0].lemmas()[0].name()
                if synonym.lower() != word.lower():  # Avoid replacement if synonym is the same as word
                    word = synonym

        # Introduce a minor typo with a 20% chance
        if random.random() < 0.2:
            if len(word) > 2:  # Only introduce typos in words with more than 2 characters
                typo_index = random.randint(0, len(word) - 1)
                typo_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:typo_index] + typo_char + word[typo_index + 1:]

        new_words.append(word)

    transformed_text = TreebankWordDetokenizer().detokenize(new_words)
    example["text"] = transformed_text
    return example

