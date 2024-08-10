import sys
#sys.path.append("../lib")
from Utils import read_fasta
import argparse

parser = argparse.ArgumentParser(description="input parameters")
parser.add_argument("-f", "--fasta", type=str, required=True, help="sequences in the fasta format")
parser.add_argument("-o", "--output", type=str, required=True, help="output file")

args = parser.parse_args()
fasta = args.fasta
output = args.output


idx, seq = read_fasta(fasta)
with open(output, "w") as w:
    for s in seq:
        w.write(s + " ")
