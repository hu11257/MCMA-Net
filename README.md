# MCMA-Net: a multimodal deep learning model for transcription factor binding site prediction

MCMA-Net is a multimodal deep learning model for transcription factor binding site prediction. The model integrates DNA sequence information, DNA shape descriptors, and evolutionary conservation scores through a cross-modal attention framework.

This repository provides inference code for running MCMA-Net on a processed example test dataset with trained model weights.

## Test dataset and model weights

A processed example test dataset and the corresponding trained model weights are available on Zenodo:

**Zenodo record:** https://zenodo.org/records/20741528
**DOI:** https://doi.org/10.5281/zenodo.20741528

The Zenodo record contains:

```text
MCMA-Net_example_test_data.zip
MCMA-Net_example_checkpoint.zip
```

The provided example data and model weights allow users to run MCMA-Net on a processed test dataset without retraining the model.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n mcma python=3.10 -y
conda activate mcma

pip install numpy pandas scikit-learn torch
```

Required Python packages:

```text
numpy
pandas
scikit-learn
torch
```

CPU-only inference is supported for the provided example test dataset. If a GPU environment is used, please install the PyTorch version that matches the local CUDA configuration.

## Download and prepare the example data

First, clone this repository:

```bash
git clone https://github.com/hu11257/MCMA-Net.git
cd MCMA-Net
```

Then download the following two files from Zenodo:

```text
MCMA-Net_example_test_data.zip
MCMA-Net_example_checkpoint.zip
```

After downloading, extract both zip files into the root directory of this repository.

The final directory structure should be:

```text
MCMA-Net/
├── README.md
├── run_test.py
├── script/
│   ├── __init__.py
│   ├── dataloader.py
│   └── model.py
├── Dataset/
│   └── example_TF/
│       ├── Sequence/
│       │   └── Test_seq.csv
│       ├── Shape/
│       │   ├── Test_HelT.csv
│       │   ├── Test_MGW.csv
│       │   ├── Test_ProT.csv
│       │   ├── Test_Roll.csv
│       │   └── Test_EP.csv
│       └── conservation/
│           └── Test_Conservation.csv
└── checkpoints/
    └── example_TF/
        └── model_final.pth
```

## Run MCMA-Net on the example test dataset

Run the following command:

```bash
python run_test.py \
  --data_dir Dataset/example_TF \
  --checkpoint checkpoints/example_TF/model_final.pth \
  --out_dir result/example_TF
```

This command loads the processed example test dataset, loads the trained model weights, performs inference, and saves the prediction results and evaluation metrics.

## Input format

MCMA-Net requires three aligned inputs for each 101 bp DNA sequence:

1. DNA sequence input
2. DNA shape input
3. Evolutionary conservation input

The rows in all input files must follow the same sample order.

### 1. DNA sequence input

File path:

```text
Dataset/example_TF/Sequence/Test_seq.csv
```

`Test_seq.csv` contains DNA sequences and binary labels.

The expected format is:

```text
sample_id, sequence, label
```

The second column is the DNA sequence, and the third column is the binary label.

Labels are defined as:

```text
1: TFBS-containing sequence
0: non-binding sequence
```

Each DNA sequence is expected to have a length of 101 bp. Non-standard or ambiguous bases are encoded as zero vectors during preprocessing.

### 2. DNA shape input

File paths:

```text
Dataset/example_TF/Shape/Test_HelT.csv
Dataset/example_TF/Shape/Test_MGW.csv
Dataset/example_TF/Shape/Test_ProT.csv
Dataset/example_TF/Shape/Test_Roll.csv
Dataset/example_TF/Shape/Test_EP.csv
```

These five files contain the DNA shape descriptors used by MCMA-Net:

```text
HelT: helical twist
MGW: minor groove width
ProT: propeller twist
Roll: roll
EP: electrostatic potential
```

Each file is a CSV matrix. Each row corresponds to one sample, and each column corresponds to one nucleotide position.

The five DNA shape descriptors are stacked as five input channels.

### 3. Evolutionary conservation input

File path:

```text
Dataset/example_TF/conservation/Test_Conservation.csv
```

This file contains phyloP100way conservation scores.

Each row corresponds to one sample, and each column corresponds to one nucleotide position.

The conservation scores are concatenated with the five DNA shape descriptors to form a six-channel physical descriptor input.

## Output format

After running `run_test.py`, the output directory will contain:

```text
result/example_TF/
├── predictions.csv
└── metrics.csv
```

### `predictions.csv`

This file contains sample-level predictions.

Columns:

```text
sample_index, sequence, true_label, probability, prediction
```

Column descriptions:

```text
sample_index: index of the test sample
sequence: DNA sequence
true_label: ground-truth binary label
probability: predicted probability of the positive TFBS class
prediction: binary prediction using a threshold of 0.5
```

### `metrics.csv`

This file contains evaluation metrics on the example test dataset.

Columns:

```text
accuracy, roc_auc, pr_auc
```

Column descriptions:

```text
accuracy: classification accuracy
roc_auc: area under the receiver operating characteristic curve
pr_auc: area under the precision-recall curve
```

## Example workflow

A complete example workflow is:

```bash
git clone https://github.com/hu11257/MCMA-Net.git
cd MCMA-Net

conda create -n mcma python=3.10 -y
conda activate mcma

pip install numpy pandas scikit-learn torch
```

Download and extract the two files from Zenodo:

```text
https://zenodo.org/records/20741528
```

Then run:

```bash
python run_test.py \
  --data_dir Dataset/example_TF \
  --checkpoint checkpoints/example_TF/model_final.pth \
  --out_dir result/example_TF
```

The expected output files are:

```text
result/example_TF/predictions.csv
result/example_TF/metrics.csv
```
