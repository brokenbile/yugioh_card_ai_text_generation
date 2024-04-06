from flask import Flask, request, render_template

import transformers
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.models.gpt2 import GPT2Model, GPT2Tokenizer

app = Flask(__name__)
#load model as global variable
#disable concurrency mode
#have function that generates text
#good luck :)

from transformers import AutoTokenizer



def tokenize_function(examples):
    return tokenizer(examples['card info'])

from transformers import AutoModelForCausalLM


def generate_text(input_text):
    model_checkpoint = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    if torch.cuda.is_available():
        device = "cuda:0" 
    else:
        device = "cpu"
    print(device)

    model = AutoModelForCausalLM.from_pretrained("ai_model/.")
    
    text_string = "/// " + input_text

    from transformers import pipeline
    generator = pipeline('text-generation', model = model, tokenizer = tokenizer)
    output = generator(text_string, max_length = 300, num_return_sequences=1)

    for output_sequence in output:
        output_string = output_sequence['generated_text']
        return output_string.split('...')[0][4:]

@app.route('/')
def my_form():
    return render_template('test.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = generate_text(text)
    return processed_text

