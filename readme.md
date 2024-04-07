# Yu-Gi-Oh Card Generation AI Model
> An AI model utilising Transformers to generate card text

### Setup

Install the required packages:

```
pip install "transformers[torch]"
pip install datasets

pip install torch --index-url https://download.pytorch.org/whl/cu118
or
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Information
This is an AI model which utilises Transformers to generate card information for Yu-Gi-Oh Cards, with files to either fine-tune a model using a dataset constructed with card information, or load a pre-trained model to generate text as prompted. Please note that other

yugioh_card_info.csv contains almost every Yu-Gi-Oh card released, with each card containing information including card name, the description of the card, the type of card, and other relevant information depending on the type of card. Every card is represented using the same format, with '--' separators to separate information. '///' indicates the start of the card's description, and '...' indicates the end of the description

Examples of how cards are represented as text strings:
* "/// Blue-Eyes White Dragon -- Normal Monster -- This legendary dragon is a powerful engine of destruction. Virtually invincible, very few have faced this awesome creature and lived to tell the tale -- 3000 ATK -- 2500 DEF -- LEVEL 8 -- Dragon -- LIGHT ..."
* "/// Polymerization -- Spell Card -- Fusion Summon 1 Fusion Monster from your Extra Deck, using monsters from your hand or field as Fusion Material -- Normal Spell ..."

All card descriptions were obtained from https://db.ygoprodeck.com/api/v7/cardinfo.php

train_ai_model.py trains a distilgpt2 model by fine-tuning, saving the fine-tuned model to the 'ai_model/' directory.

load_ai_model.py loads a pre-trained model stored in the 'ai_model/' directory, allowing the user to generate text as prompted. By default, 5 separate strings are generated based on the user's input

Additionally, server.py contains code for a flask server, loading a webpage which takes a user's input in a text box, and outputs a single generated text string in a new webpage.
The html file for the flask server is located in templates/test.html, which can be changed as needed