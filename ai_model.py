import transformers
import datasets
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer

from datasets import load_dataset
dataset = load_dataset('csv', data_files='yugioh_card_info_2.csv')

datasets = dataset['train'].train_test_split(test_size=0.1)

from transformers import AutoTokenizer
model_checkpoint = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples['card info'])

tokenized_datasets = datasets.map(tokenize_function, num_proc=8, remove_columns=["card info"])

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

tokenizer.decode(lm_datasets["train"][1]["input_ids"])

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

from transformers import Trainer, TrainingArguments

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-yugioh",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=20,
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
)

trainer.train()

text_string = input("Enter text:")

from transformers import pipeline
generator = pipeline('text-generation', model = model, tokenizer = tokenizer, device="cuda")
generator(text_string, max_length = 300, num_return_sequences=3)