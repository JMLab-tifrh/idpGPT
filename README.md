# idpGPT
GPT Protein Language Models to generate novel protein sequences.</br>
Three PLMs have been trained. </br>
LLPS+ GPT : generates sequences highly prone to liquid liquid phase separation (LLPS). model saved  as llps_plus_gpt.pt.</br>
LLPS- GPT : generates sequences which can undergo LLPS but with leser intensity than LLPS+. model saved as llps_minus_gpt.pt.</br>
PDB* GPT  : generates sequences that would not undergo LLPS unless very drastic conditions are applied. model saved as no_PS.pt</br>

## Usage
please set the following environment variable to use the scripts </br>
bash : `export PYTHONPATH=$PYTHONPATH:<path to the lib directory>`</br>
csh  : `setenv PYTHONPATH "$PYTHONPATH:<path to the lib directory>`</br>
zsh  : `export PYTHONPATH=$PYTHONPATH:<path to the lib directory>`</br>
fish : `set -x PYTHONPATH $PYTHONPATH <path to the lib directory>`</br>

`lib` is the directory present in this repository. it contains libraries used to train PLMs and generate sequences.</br>

The above variable would need to be set everytime a terminal is opened.</br>
Hence better way is to put the line in the respective configuration file.</br>
bash : `~/.bashrc`</br>
csh  : `~/.cshrc`</br>
zsh  : `~/.zshrc`</br>
fish : `~/.config/fish/config.fish`</br>

after including the environment variable, please source the respective configuration file. </br>
For example, in bash, `source ~/.bashrc`.</br>

Each script has its usage instructions. type `python <script> -h` to see help.</br>
Eg. `python train_gpt.py -h'</br>

## The output format when using the sequence generator
The sequences generated are saved in a the fasta format which is a standard format to save protein sequence(s).</br>
Fasta files are plain text files and generally have the extension .fa or .fasta. Below is an example of a file in fasta format</br>
```
>prot1
RGGAFGGKLVFFSSRGG
>prot 2
MAVCQYPLVVQQK
```
The line(s) starting with ">" contains the sequences identifier(s). It does not have to be unique for each sequence, but the fasta format requires an identifier.
Below an identifier and before the next identifier, a protein sequence will be written. For long protein sequences, the sequences can be decomposed into multiple
lines for readability. An example for a longer dummy sequence is given below:
```
>dummy sequence (95 residues)
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
> a shorter sequence
VVVVVVVVV
```

## Requirements
- numpy
- openMM
- scikit-learn
- pytorch
- cuda
- MDAnalysis
  

## Disclaimer
We have used the codes in https://github.com/Infatoshi/fcc-intro-to-llms as a reference for our codes.
