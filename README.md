# Leveraging Information Theory to Select Host-Virus Testing Pairs for Cross-Species Transmission

This is a repository for Cross-Species Transmission (CST) testing pair selection utilizing entropy and information theory. It is used to generate figures and data sets for RECOMB2025 submission.

## Requirements
Use requirements.txt to check dependencies and Python packages. 

```
pip install -r /path/to/requirements.txt
```
## Parameters
There are multiple parameters for the script with optional arguments. Filename argument takes networkx pickle file 
```
$ python3 entropy_calc_node_index.py -h
usage: entropy_calc_node_index.py [-h] [--n N] [--filename FILENAME]
                                  [--texa TEXA] [--est EST]

Input argument for plotting

optional arguments:
  -h, --help           show this help message and exit
  --n N                Number of edges to be added
  --filename FILENAME  networkx pickle file for input network
  --texa TEXA          networkx pickle file for texanomic network
  --est EST            estiamtion of entropy

```
