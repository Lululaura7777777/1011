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
    # Tokenize the text into words
    words = word_tokenize(example["text"])
    transformed_words = []

    for word in words:
        # 随机选择是否对该词进行转换（例如，20%的概率）
        if random.random() < 0.2:
            # 随机选择进行同义词替换或拼写错误
            if random.random() < 0.5:
                # 同义词替换
                synsets = wordnet.synsets(word)
                if synsets:
                    synonyms = synsets[0].lemma_names()  # 获取同义词
                    if synonyms:
                        synonym = random.choice(synonyms)  # 随机选择一个同义词
                        transformed_words.append(synonym)
                    else:
                        transformed_words.append(word)  # 没有同义词，保留原词
                else:
                    transformed_words.append(word)  # 没有找到同义词，保留原词
            else:
                # 拼写错误模拟
                if len(word) > 1:
                    typo_word = list(word)
                    random_idx = random.randint(0, len(word) - 1)  # 随机选择一个字符
                    # 用键盘相邻字符代替 (简单示例，仅适用于少数字母)
                    typo_replacements = {'a': 's', 'e': 'r', 'i': 'o', 'o': 'i', 'u': 'y'}
                    typo_word[random_idx] = typo_replacements.get(word[random_idx], word[random_idx])
                    transformed_words.append("".join(typo_word))
                else:
                    transformed_words.append(word)  # 单字符不改变
        else:
            # 不进行转换
            transformed_words.append(word)

    # Detokenize the transformed words back into a string
    example["text"] = TreebankWordDetokenizer().detokenize(transformed_words)
    return example

