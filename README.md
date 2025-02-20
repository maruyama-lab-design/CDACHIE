# CDACHIE

## Abstract
This repository contains the code for the paper "CDACHIE: Chromatin Domain Annotation by Integrating Chromatin Interaction and Epigenomic Data with Contrastive Learning."

CDACHIE (Chromatin Domain Annotation using Contrastive Learning for Hi-C and Epigenomic Data) is a method for identifying chromatin domains from Hi-C and epigenomic data. Our approach leverages contrastive learning to generate aligned representative vectors for both data types at each genomic bin. The concatenated vectors are then clustered using $K$-means to classify distinct chromatin domain types.


## Installation
1. Clone the repository
```bash
git clone https://github.com/maruyama-lab-design/CDACHIE.git
```
2. Install the required packages in the environment
```bash
pip install -r requirements.txt
```
**Note:** The requirements.txt file is still being prepared and will be added soon.

## Data
We used data from the GM12878 cell line for our experiments.  
The data directory structure is as follows:
```
data
├── cluster
│    ├── CDACHIE.bed
│    └── clusters.csv
├── feature
│    ├── GM12878_100000_bins.txt
│    ├── *.bigwig
│    └── signals_1kb.npy
└── hic_embedding
     └── hic_line_embedding_128.csv
```

### Hi-C embedding
The Hi-C embedding data file (hic_line_embedding_128.csv) was created using the code from: https://github.com/nedashokraneh/IChDA/blob/master/src/dataset_class.py

### Epigenomic signal data
First, download the epigenomic data bigwig files, then run `make_feature.ipynb` to create the epigenomic signal data file (`signals_1kb.npy`).

## Contrastive Learning & Clustering
Set the hyperparameters in `config.yaml`, then run:
```bash
python main.py
```
The annotation results will be saved to `data/cluster/clusters.csv`.

## Usage
To use CDACHIE for your own data:
1. Prepare your Hi-C and epigenomic data in the required format
2. Modify the configuration in `config.yaml` as needed
3. Run the main script
4. The resulting chromatin domain annotations will be available in the output files
