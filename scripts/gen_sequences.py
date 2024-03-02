import numpy as np
import torch
from tqdm.auto import trange, tqdm
import pickle
import sys
sys.path.append("lib")
from ModelClasses import *
from Helpers import *
from Utils import *


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torch.load("models/no_PS_GPT.pt")

with open('data/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.read()
    chars = sorted(set(vocab))
vocab_size = len(chars)

string2int = { ch:i for i,ch in enumerate(chars) }
int2string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string2int[c] for c in s]
decode = lambda l: ''.join([int2string[i] for i in l])

# generate sequences for validation
N = 100
min_len = 60
max_len = 70

with open("/home/wasim/idp-gpt/sims/no_PS/generated_sequences.fa", "w") as w:
    pbar = trange(N)
    for i in range(N):
        prompt = vocab[np.random.randint(len(vocab))]
        length = np.random.randint(low=min_len, high=max_len+1)
        context = torch.tensor(encode(prompt), dtype=torch.long, device="cuda")
        generated_chars = decode(model.generate(context.unsqueeze(0),
                                                max_new_tokens=length-1, block_size=128,
                                                device=device)[0].tolist())
        try:
            w.write(f">prot{i:d}\n"+replace_res(generated_chars)+"\n")
            pbar.update()
        except:
            i -= 1
        
