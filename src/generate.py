import torch
from model import GPTLanguageModel
from train import decode, encode 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# start text
start_text = "Hello World"
# convert to token IDs
start_ids = torch.tensor([encode(start_text)], dtype=torch.long, device=device) 
# shape will be (1, len(start_text))
# now pass this as context
context = start_ids
# generate from the model
model = GPTLanguageModel()
m = model.to(device)
m.eval() # put the model in inference mode


print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
# If you want to load a pretrained model from disk instead, uncomment this:
# model = GPTLanguageModel()   # must match the architecture you trained
# model.load_state_dict(torch.load("/__pycache__/gpt_model.pth"))
# model.eval()  # put into inference mode
# m = model.to(device)
