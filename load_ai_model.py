import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer

from datasets import load_dataset

if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    if torch.cuda.is_available():
        device = "cuda:0" 
    else:
        device = "cpu"

    def tokenize_function(examples):
        return tokenizer(examples['card info'])

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

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("ai_model/.")

    def generate_text(input_text):
        text_string = "/// " + input_text

        from transformers import pipeline
        generator = pipeline('text-generation', model = model, tokenizer = tokenizer, device="cuda")
        output = generator(text_string, max_length = 300, num_return_sequences=3)

        for output_sequence in output:
            output_string = output_sequence['generated_text']
            print(output_string.split('...')[0])

    print(generate_text("test 1"))
    print(generate_text("Infernoble Knight "))