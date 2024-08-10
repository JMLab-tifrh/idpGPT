from math import log2
import numpy as np
from itertools import islice


def read_fasta(file):
    sequences = []
    entry = []
    seq = ""

    with open(file, "r") as f:
        line = f.readline()
        
        while line:
            if line[0] == ">":
                sequences.append(seq)
                entry.append(line[1:-1])
                seq = ""
                line = f.readline()
                continue

            seq += line[:-1]
            line = f.readline()
        sequences.append(seq)
    return entry, sequences[1:]


def split2fasta(seq, deffnm, path, max_len=500):
    r"""seq it the a list of the seuqnces, deffnm is the default
filename, path is the path where the files will be saved,
max_len is the maximum number of sequences a file will contain"""
    
    split = [seq[i*max_len:(i+1)*max_len] for i in range(len(seq)//max_len)]
    
    ctr = 0
    for i, s in enumerate(split):
        with open(f"{path}/{deffnm}{i}.fasta", "w") as w:
            for seq in s: 
                seq = remove_spaces(seq)
                w.write(f">prot{ctr}\n{seq}\n")
                ctr += 1
                
                
def remove_spaces(s):
    s = list(s)
    while ' ' in s:
        s.remove(" ")
    return ''.join(s)


def replace_res(seq, U="C"):
    chars = np.array(list(seq))
    chars[np.where(chars == "U")] = U

    chars = list(chars)
    while "X" in chars:
        chars.remove("X")
    while " " in chars:
        chars.remove(" ")

    return "".join(chars)


def KDH(seq, mean=True):
    """calculate Kyte-Doolittle hydropathy"""
    KD = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
          'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
          'R': -4.0, 'S': -0.8, 'T': -0.7, 'U': 2.5, 'V': 4.2, 'W': -0.9, 'X': 0.0,
          'Y': -1.3}
    h = sum([KD[s] for s in remove_spaces(seq)])
    if mean:
        return h/len(seq)
    return h


def calc_entropy(sequence, vocab=None):
    if sequence is None:
        return None
    
    sequence = remove_spaces(sequence)
    
    if vocab is None:
        vocab = ['A', 'G', 'I', 'L', 'P', 'V', 'F', 'W', 'Y', 'D', 'E',
                 'R', 'H', 'K', 'S', 'T', 'C', 'M', 'N', 'Q']

    count = {s:0 for s in vocab}
    for s in sequence:
        count[s] += 1/len(sequence)
    entropy = sum([p*log2(p) if p > 0 else 0 for p in list(count.values())])

    return -entropy


def seg(protein_sequence, window_size=12, threshold=2.2):
    lcr_regions = []
    
    # Iterate over the protein sequence with a sliding window
    for i in range(len(protein_sequence) - window_size + 1):
        window = protein_sequence[i:i+window_size]
        
        # Calculate the complexity score for the window
        complexity_score = calc_entropy(window)
        
        lcr_regions.append(1 if complexity_score < threshold else 0)
    
    return lcr_regions



class StandardNormal:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        
    def transform(self, data):
        return (data - self.mean)/self.std
    
    def invert(self, data):
        return (data*self.std) + self.mean
    
class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None
        
    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        
    def transform(self, data):
        return (data - self.min)/(self.max - self.min)
    
    def invert(self, data):
        return data*(self.max - self.min) + self.min
