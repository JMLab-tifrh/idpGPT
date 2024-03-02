import numpy as np
import torch
import torch.nn as nn
import mmap
import random

def get_random_chunk(filename, encode, decode, block_size, batch_size):
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)    

    return data

def get_batch(filename, encode, decode, batch_size, block_size, device=None):
    data = get_random_chunk(filename, encode, decode, block_size, batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    x, y = x.to(device), y.to(device)
    return x, y


def make_batch(data, batch_size=256):
    n_batch = int(np.ceil(data.shape[0]//batch_size))
    return [data[i*batch_size:(i+1)*batch_size] for i in range(n_batch+1)]


def get_token_embedding(model, vocab, seq, device):
    chars = sorted(set(vocab))
    vocab_size = len(chars)

    string2int = { ch:i for i,ch in enumerate(chars) }
    int2string = { i:ch for i,ch in enumerate(chars) }

    encode = lambda s: [string2int[c] for c in s]
    decode = lambda l: ''.join([int2string[i] for i in l])
    
    encoded = torch.tensor(encode(seq), device=device)
    
    with torch.no_grad():
        token_embd = model.token_embedding_table(encoded)
        pos_embd = model.position_embedding_table(encoded)
    
    return torch.mean(token_embd + pos_embd, dim=0).cpu().numpy()