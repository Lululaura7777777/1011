import datasets
from datasets import load_dataset, concatenate_datasets
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os

# Set seed for reproducibility
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenization function
def tokenize_function(examples):
    if "text" in examples:
        text_key = "text"
    elif "sentence" in examples:
        text_key = "sentence"
    else:
        raise KeyError("Neither 'text' nor 'sentence' found in dataset examples. Please check the dataset structure.")
    
    return tokenizer(examples[text_key], padding="max_length", truncation=True)



# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    model.train()
    model.to(args.device)
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Move to device, handling only tensors
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass and optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update progress
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

    print("Training completed.")
    model.save_pretrained(save_dir)

# Core evaluation function
def do_eval(eval_dataloader, model_dir, device, out_file):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    metric = evaluate.load("accuracy")
    with open(out_file, "w") as out_f:
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

            # Write predictions and labels to file
            for pred, label in zip(predictions, batch["labels"]):
                out_f.write(f"{pred.item()}\n{label.item()}\n")

    return metric.compute()

# Create augmented dataloader
def create_augmented_dataloader(args, dataset):
    random_transformed_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
    transformed_dataset = random_transformed_dataset.map(custom_transform, load_from_cache_file=False)

    # Ensure labels are correctly named
    if "label" in dataset["train"].column_names:
        dataset["train"] = dataset["train"].rename_column("label", "labels")
    if "label" in transformed_dataset.column_names:
        transformed_dataset = transformed_dataset.rename_column("label", "labels")

    combined_dataset = concatenate_datasets([dataset["train"], transformed_dataset])
    train_dataloader = DataLoader(combined_dataset, shuffle=True, batch_size=args.batch_size)

    return train_dataloader

# Create dataloader for transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    if debug_transformation:
        sample_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        sample_transformed = sample_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print(f"Original: {sample_dataset[k]}\nTransformed: {sample_transformed[k]}")
        exit()

    transformed_dataset = dataset["test"].map(custom_transform, load_from_cache_file=False)
    tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    if "label" in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    tokenized_dataset.set_format("torch")
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

    return eval_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train the model on training data")
    parser.add_argument("--train_augmented", action="store_true", help="train model on augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate the model on test data")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on transformed test data")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true", help="use subset for debugging training loop")
    parser.add_argument("--debug_transformation", action="store_true", help="print transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    if args.debug_train:
        train_dataloader = DataLoader(small_train, shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(small_eval, batch_size=args.batch_size)
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size)

    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        do_train(args, model, train_dataloader, save_dir=args.model_dir)

    if args.train_augmented:
        augmented_dataloader = create_augmented_dataloader(args, tokenized_dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        do_train(args, model, augmented_dataloader, save_dir=f"{args.model_dir}_augmented")

    if args.eval:
        score = do_eval(eval_dataloader, args.model_dir, args.device, out_file=f"{args.model_dir}_original.txt")
        print("Evaluation Score:", score)

    if args.eval_transformed:
        transformed_dataloader = create_transformed_dataloader(args, tokenized_dataset, args.debug_transformation)
        score = do_eval(transformed_dataloader, args.model_dir, args.device, out_file=f"{args.model_dir}_transformed.txt")
        print("Transformed Evaluation Score:", score)


