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

## Requirements
- numpy
- openMM
- scikit-learn
- pytorch
- cuda
- MDAnalysis
  

## Disclaimer
We have used the codes in https://github.com/Infatoshi/fcc-intro-to-llms as a reference for our codes.
