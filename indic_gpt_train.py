import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import os.path

from gpt_model import GPTLanguageModel 

# अतिप्राचल
# Hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 
eval_interval = 10
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

# संचिका पथ
# File paths
input_text_path = 'hi.txt'
tokenizer_path = 'saved_tokenizer' 
model_path = 'saved_model' 
# ------------

torch.manual_seed(1337)

# यदि सहेजा गया संकेतक उपलब्ध है, तो उसको प्रप्त करें। नहीं तो प्रारंभ से प्रशिक्षित करें।
# If a saved tokenizer is available, then retrieve it. If not, then train from scratch.
if os.path.isfile(tokenizer_path):
  print('Pre-trained tokenizer file detected. Loading...')
  tokenizer = Tokenizer.from_file(tokenizer_path)
  print('Tokenizer loaded!')
else:
  tokenizer = Tokenizer(BPE())
  tokenizer.pre_tokenizer = Whitespace()
  trainer = BpeTrainer(vocab_size=10000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
  tokenizer.train(files=[input_text_path], trainer=trainer)
  tokenizer.save(tokenizer_path)

# शब्दावली का परिमाण प्राप्त करें।
# Retrieve the vocabulary size.
vocab_size = tokenizer.get_vocab_size()

# आंकड़ा संचिका खोलें और पाठ पढ़ें।
# Open the data file and read the text.
with open(input_text_path, 'r', encoding='utf-8') as f:
  text = f.read()

# आंकड़ों को संकेतक द्वारा कूटलेखित कर के torch.tensor में परिवर्तित करें।
# Use the tokenizer to encode the data and convert it to a torch.tensor.
data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
del text

# प्रशिक्षण और परीक्षण विभाजन
# Train and test splits
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# आंकड़ा भारण
# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(vocab_size)
print('Defining dummy model...')
# यदि सहेजा गया संकेतक उपलब्ध है, तो उसको प्रप्त करें।
# If a saved tokenizer is available, then retrieve it.
if os.path.isfile(model_path):
  model.load_state_dict(torch.load(model_path))
  print('Loaded model.')
m = model.to(device)
print("Moved model to cuda device")

# PyTorch अनुकूलक बनाएं।
# Create a PyTorch optimizer.
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

losses = estimate_loss()
best_val = losses['val']

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val:
            print("Saving new best model...")
            torch.save(m.state_dict(), 'saved_model')
            best_val = losses['val']

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

