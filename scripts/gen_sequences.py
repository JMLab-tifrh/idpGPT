import numpy as np
import torch
from tqdm.auto import trange, tqdm
import pickle
import sys
import argparse
from ModelClasses import *
from Helpers import *
from Utils import *


# reading arguments
parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("-m", "--model", type=str, required=True, help="pre-trained model")
parser.add_argument("-N", "--num", type=str, required=True, help="number of sequences to generate")
parser.add_argument("-min", "--min-len", type=str, required=True, help="minimum length of the sequence(s)")
parser.add_argument("-max", "--max-len", type=str, required=True, help="maximum length of the sequence(s)")
parser.add_argument("-voc", "--vocab", type=str, required=True, help="vocabulary use to train the model")
parser.add_argument("-o", "--output", type=str, required=True, help="output FASTA file (.fa/.fasta)")
parser.add_argument("-d", "--device", type=str, required=False, default="auto",
                    help="device to run the model on (cpu/gpu/auto). [default=auto]")

args = parser.parse_args()

model_file = args.model
vocab_file = args.vocab
N = int(args.num)
min_len = int(args.min_len)
max_len = int(args.max_len)
output = args.output
dev = args.device

if dev == "auto":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
elif dev == "gpu":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("GPU selected, but not available. switching device to cpu.")
        device = torch.device("cpu")
elif dev == "cpu":
        device = torch.device("cpu")
else:
    print("wrong input for device. exiting.")
    exit(0)

model = torch.load(model_file).to(device)

with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab = f.read()
    chars = sorted(set(vocab))
vocab_size = len(chars)

string2int = { ch:i for i,ch in enumerate(chars) }
int2string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string2int[c] for c in s]
decode = lambda l: ''.join([int2string[i] for i in l])

lengths = []
with open(output, "w") as w:
    pbar = trange(N)
    for i in range(N):
        prompt = vocab[np.random.randint(len(vocab))]
        length = np.random.randint(low=min_len, high=max_len+1)
        context = torch.tensor(encode(prompt), dtype=torch.long, device="cuda")
        generated_chars = decode(model.generate(context.unsqueeze(0),
                                                max_new_tokens=length-1, block_size=128,
                                                device=device)[0].tolist())
        try:
            corrected_seq = replace_res(generated_chars)
            w.write(f">prot{i:d}\n"+ corrected_seq +"\n")
            pbar.update()
            lengths.append(len(corrected_seq))
        except:
            i -= 1
       
print(f"""generated sequences saved to {output}.
Minimum length of generated sequences = {min(lengths):d}
Maximum length of generated sequences = {max(lengths):d}""")
