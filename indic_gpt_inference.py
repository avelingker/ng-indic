import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

import os.path

from gpt_model import GPTLanguageModel 

# संचिका पथ 
# File paths
tokenizer_path = 'saved_tokenizer' 
model_path = 'saved_model' 
output_text_path = 'generated_output'
# ------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# सहेजा गया संकेतक प्राप्त करें। यदि उपलब्ध नहीं तो समाप्त करें।
# Get the saved tokenizer. If not available, then exit.
if os.path.isfile(tokenizer_path):
  print('Pre-trained tokenizer file detected. Loading...')
  tokenizer = Tokenizer.from_file(tokenizer_path)
  print('Tokenizer loaded!')
else:
  print('Pre-trained tokenizer not found. Exiting.')
  exit()

# शब्दावली का परिमाण प्राप्त करें।
# Retrieve the vocabulary size.
vocab_size = tokenizer.get_vocab_size()

model = GPTLanguageModel(vocab_size)
print('Defining dummy model...')

# सहेजा गया भाषा प्रतिरूप प्राप्त करें। यदि उपलब्ध नहीं तो समाप्त करें।
# Get the saved language model. If not available, then exit.
if os.path.isfile(model_path):
  model.load_state_dict(torch.load(model_path))
  print('Loaded model.')
else:
  print('Pre-trained model not found. Exiting.')
  exit()
m = model.to(device)
print("Moved model to cuda device")

# प्रशिक्षित भाषा प्रतिरूप से पाठ उत्पन्न करें।
# Generate text from the trained language model.
with open(output_text_path, 'w') as f:
  idx = torch.zeros((1,1), dtype=torch.long, device=device)
  f.write(tokenizer.decode(m.generate(idx , max_new_tokens=2000)[0].tolist()))
