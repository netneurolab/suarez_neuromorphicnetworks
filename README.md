# Learning function from structure in neuromorphic networks


## "What's in this repository?"

This repository contains code for the manuscript "[Learning function from structure in neuromorphic networks](https://www.biorxiv.org/content/10.1101/2020.11.10.350876v1)" by Laura Suarez, Blake Richards, Guillaume Lajoie & Bratislav Misic.

We investigated the link between macroscale connectivity and the computational properties that emerge from network dynamics in the human connectome.
We've tried to document the various aspects of this repository with this README file, so feel free to check things out.

## "How to run the things from scratch?"

First, you'll need to make sure you have installed the appropriate software packages, and have downloaded the appropriate data files. 

1. Git clone the [suarez_neuromorphicnetworks](https://github.com/estefanysuarez/neuromorphic-networks) repository.
2. Download the "data" folder from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4311814.svg)](https://doi.org/10.5281/zenodo.4311814), and place this folder into the repository's root directory!"
3. In the command line, type:

```bash
cd suarez_neuromorphicnetworks
conda env create -f environment.yml
conda activate rcomputing
```

4. Git clone [reservoir](https://github.com/estefanysuarez/reservoir.git) repository.
5. And install it in the newly created environment:
```bash
cd reservoir
pip install .
```

4. Then, follow these steps sequentially (i.e., first "Run simulations", then "Compile results" and then "Analyses and Figures"). 

### Step 1. Run simulations
```bash
python scripts/01_rc_workflow/1_run_rc_workflow.py

python scripts/01_rc_workflow/2_get_network_properties.py
```

### Step 2. Compile results
```bash
python scripts/01_fetch_results/fetch_task_results.py

python scripts/01_rc_workflow/fetch_net_props_results.py
```
### Step 3. Analyses and figures
```bash
python scripts/03_analysis/figX.py
```
(replace X by the number of the figure)

## "How to run only the analyses?"
1. Git clone the [suarez_neuromorphicnetworks](https://github.com/estefanysuarez/neuromorphic-networks) repository.
2. Download the "data" and "proc_results" folders from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4311814.svg)](https://doi.org/10.5281/zenodo.4311814), and place them into the repository's root directory!"

To run the analysis presented in Figure "X" of the manuscript, you just need to run:

```bash
python scripts/03_analysis/figX.py
```

## "I have some questions..."

[Open an issue](https://github.com/estefanysuarez/neuromorphic-networks/issues) on this repository and someone will try and get back to you as soon as possible!
