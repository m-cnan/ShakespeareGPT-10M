import torch
from model import GPTLanguageModel
import os

# Load the data to get vocab_size and create encode/decode functions
with open('../data/little-shakespear/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# start text
start_text = "Hello World"
# convert to token IDs
start_ids = torch.tensor([encode(start_text)], dtype=torch.long, device=device) 
# shape will be (1, len(start_text))
# now pass this as context
context = start_ids
# Load the trained model
model = GPTLanguageModel(vocab_size)
model_path = '../__pycache__/gpt_model.pth'

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded trained model from", model_path)
else:
    print("No trained model found. Please run train.py first.")

m = model.to(device)
m.eval() # put the model in inference mode

print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
