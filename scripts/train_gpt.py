import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import trange, tqdm
import mmap
import random
import pickle
import sys
import argparse
from ModelClasses import *
from Helpers import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# hyper-parameters
## touch these if you know exactly what you are doing
block_size = 128
batch_size = 64
n_embd = 256
n_head = 8
n_layers = 8
learning_rate = 3e-4
eval_iters = 10
eval_interval = 1000
dropout = 0.2

# functions
@torch.no_grad()
def estimate_loss(model, train, val, block_size, batch_size, eval_iters=100):
    out = {}
    model.eval()
    
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(train, encode, decode, block_size=block_size, batch_size=batch_size)
        logits, loss = model(X, Y, device=device)
        losses[k] = loss.item()
    out["train_loss"] = losses.mean()

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val, encode, decode, block_size=block_size, batch_size=batch_size)
        logits, loss = model(X, Y, device=device)
        losses[k] = loss.item()
    out["val_loss"] = losses.mean()    
    
    model.train()
    return out


# reading arguments
parser = argparse.ArgumentParser(description="Input parameters.")
parser.add_argument("-v", "--vocab", type=str, required=True, help="vocabulary file")
parser.add_argument("-s", "--seq", required=True, type=str, help="sequences prepared for input. use prepare_seq.py along with a fasta file to generate this file.")
parser.add_argument("-e", "--epochs", type=str, help="number of epochs to train", default=int(1e7))
parser.add_argument("-o", "--output", type=str, required=True, help="output path for the model")
parser.add_argument("-l", "--losses", type=str, required=True, help="output path for the losses")

args = parser.parse_args()
vocab_file = args.vocab
text_file = args.seq
epochs = int(args.epochs)
model_out = args.output
loss_out = args.losses

# execution
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab = f.read()
    chars = sorted(set(vocab))
vocab_size = len(chars)

string2int = { ch:i for i,ch in enumerate(chars) }
int2string = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [string2int[c] for c in s]
decode = lambda l: ''.join([int2string[i] for i in l])


model = GPTLanguageModel(vocab_size, n_embd, block_size,
                         n_layers, n_head, device).to("cuda")

opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)


iterator = trange(epochs)
losses = {"train_loss":[], "val_loss":[]}
iterator.set_postfix_str("train_loss=----, val_loss=-----")

for i in iterator:
    x, y = get_batch(text_file, encode, decode,
                     batch_size=batch_size, block_size=block_size)
    logits, loss = model.forward(x, y, device=device)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if i%eval_interval == 0:
        _ = estimate_loss(model,
                          train=text_file,
                          val=text_file,
                          block_size=block_size, batch_size=batch_size,
                          eval_iters=eval_iters)
    
        losses["train_loss"].append(_["train_loss"])
        losses["val_loss"].append(_["val_loss"])
        
        iterator.set_postfix_str(f"train_loss={_['train_loss']:.6f}, val_loss={_['val_loss']:.6f}")

torch.save(model, model_out)
pickle.dump(losses, open(loss_out, "wb"))
