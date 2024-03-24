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

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("ai_model/.")

    def generate_text(input_text):
        text_string = "/// " + input_text

        from transformers import pipeline
        generator = pipeline('text-generation', model = model, tokenizer = tokenizer, device="cuda")
        output = generator(text_string, max_length = 300, num_return_sequences=5)

        for output_sequence in output:
            output_string = output_sequence['generated_text']
            print(output_string.split('...')[0])

    #generate_text("test 1")
    #generate_text("Infernoble Knight ")

    input_string = input("Enter text: ")
    generate_text(input_string)