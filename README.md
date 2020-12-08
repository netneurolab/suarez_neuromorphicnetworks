# Learning function from structure in neuromorphic-networks


## "What's in this repository?"

This repository contains code and results for the manuscript "[Learning function from structure in neuromorphic networks](https://www.biorxiv.org/content/10.1101/2020.11.10.350876v1)" by Laura Suarez, Blake Richards, Guillaume Lajoie & Bratislav Misic.
%We investigated how well different null model implementations account for spatial autocorrelation in statistical analyses of whole-brain neuroimaging data.

%We've tried to document the various aspects of this repository with a whole bunch of README files, so feel free to jump around and check things out.

## "Just let me run the things!"

Itching to just run the analyses?
You'll need to make sure you have installed the appropriate software packages, have access to the HCP, and have downloaded the appropriate data files (check out our [walkthrough](https://netneurolab.github.io/markello_spatialnulls) for more details!).
Once you've done that, you can get going with the following:

```bash
git clone https://github.com/netneurolab/markello_spatialnulls
cd markello_spatialnulls
conda env create -f environment.yml
conda activate markello_spatialnulls
pip install parspin/
make all
```


## "I have some questions..."

[Open an issue](https://github.com/netneurolab/markello_spatialnulls/issues) on this repository and someone will try and get back to you as soon as possible!
