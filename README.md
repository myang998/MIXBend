# MIXBend
Employing bimodal representations to predict DNA bendability within a self-supervised pre-trained framework

# Data description
Four 'loop-seq' datasets are downloaded from [Basu el al.'s work](https://www.nature.com/articles/s41586-020-03052-3), including two benchmark datasets Random and Nucleosomal, as well as independent test sets ChrV and Tiling. The details are described as follows:
* The Random dataset comprises 12,472 randomly generated DNA sequences
* The Nucleosomal consists of 19,907 DNA sequences selected from S. cerevisiae SacCer2 genome.
* The Tiling dataset contains 82,368 sequences, which are tiled across the chromosome with a 7 bp shift. These sequences are derived from +/- 2,000 bp regions centered around the $+1$ nucleosome dyads of 576 selected genes.
* The ChrV dataset consists of 82,404 sequences, tiled with 7 bp spacing from the yeast Saccharomyces cerevisiae chromosome V.

# Folder ./pretrain_model
This package includes the SE encoder (DNABERT) and well pretrained MIXBend, which is available in [GoogleDrive](https://drive.google.com/file/d/1mvIyHsl3qTcxo6yvI-oe3ZKqDz5Eq7Aa/view?usp=sharing).

# Steps for running MIXBend
* Download the ./pretrain_model folder from the [GoogleDrive](https://drive.google.com/file/d/1mvIyHsl3qTcxo6yvI-oe3ZKqDz5Eq7Aa/view?usp=sharing). This package contains the well pretrained parameters used in MIXBend.
* Run main.py to train the model using 10-fold cross-validation.

# Requirements
* tqdm 4.63.0
* pandas 1.3.5
* numpy 1.22.0
* transformers 4.17.0
* torch 1.10.0
* scikit-learn 1.0.2
* biopython 1.79
* pyteomics 4.5.3
* sentencepiece 0.1.99