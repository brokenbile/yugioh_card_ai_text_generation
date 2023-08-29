import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')

from datasets import load_dataset, DownloadConfig

dataset = load_dataset('csv', data_files='test.csv')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

def tokenize_function(example):
    return tokenizer(example['card info'], truncation=True)

ds = dataset['train'].train_test_split(test_size=0.1)
tokenized_datasets = ds.map(tokenize_function, num_proc=8, remove_columns=["card info"]) #Make sure to remove all the existing columns

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    "test-trainer",
    per_device_train_batch_size = 12,
    per_device_eval_batch_size = 12
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
)

trainer.train()