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

## Data
We used data from the GM12878 cell line for our experiments.  
The data directory structure is as follows:
```
data
├── input
│    ├── epigenomic_feature
│    │    ├── GM12878_100000_bins.txt
│    │    ├── *.bigwig
│    │    └── signals_1kb.npy
│    └── hic_embedding
│         └── hic_line_embedding_128.csv
└── output
     ├── CDACHIE.bed
     └── clusters.csv
```

### Epigenomic signal
Before running `cdachie.py`, you must process the epigenomic data:

1. Download the required bigwig files (detailed instructions in `make_feature.ipynb`)
2. Open and run `make_feature.ipynb` in Jupyter Notebook to generate `signals_1kb.npy`
   - This preprocessing step is mandatory
   - The notebook will guide you through converting bigwig files into the required numpy format

### Hi-C embedding
The Hi-C embedding data file (hic_line_embedding_128.csv) was created using the code from: https://github.com/nedashokraneh/IChDA/blob/master/src/dataset_class.py

## Contrastive Learning & Clustering
Only proceed with this step after completing the data preprocessing steps above.

1. Configure your parameters in `config.yaml`
2. Run the script:
```bash
python cdachie.py
```
3. The annotation results will be saved to `data/output/clusters.csv`

## Usage
To use CDACHIE for your own data:
1. Prepare your Hi-C and epigenomic data in the required format
2. Modify the configuration in `config.yaml` as needed
3. Run `cdachie.py`
4. The output file contains chromatin domain annotation
