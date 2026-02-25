# NeuroFlow

<p align="center">
  <img src="Pipline.JPG" width="800" alt="项目示意图">
</p>

## Introduction
NeuroFlow is a reproducible research codebase developed for EEG-to-image generation, retrieval, and analysis. The project implements a two-task pipeline combining EEG representation learning and FlowMatching-based image generation driven by neural signals. The repository contains training and evaluation code, visualization tools, and processed datasets to reproduce the main results in the associated manuscript. The file `pipeline.jpg` shows the overall framework.

## Key Contributions
- A modular pipeline for EEG representation learning and downstream image generation.
- A FlowMatching-based conditional generator tailored for EEG embeddings.
- Extensive visualization and ablation scripts to reproduce figures and comparisons.

## Repository Structure (selected)
- `Task1/` – EEG representation learning, retrieval training scripts, visualization, and utilities.
- `Task2/` – FlowMatching prior and generation code for image synthesis from EEG embeddings.
- `requirementsCopy.txt` – Python dependency list used for experiments.
- `pipeline.jpg` – Paper framework figure (framework diagram).

## Requirements
1. Python 3.8+ (we used Anaconda environments during development).
2. Install dependencies from `requirementsCopy.txt`:

```
pip install -r requirementsCopy.txt
```

Note: If you plan to run in the provided project subfolders, install dependencies in the root so that imports resolve correctly.

## Data
Download the original dataset from https://osf.io/3jk45/overview. Alternatively, you may use the preprocessed dataset we provide via Netdisk (contact the authors). Place the processed data in the same directory structure expected by the `Task1/` and `Task2/` scripts; 

## Reproducible Runs

**Task 1 — EEG representation training and experiments**

1. Change directory to `Task1/`:

```
cd Task1/
```

2. Train / run the Task1 pipeline (training + evaluation):

```
python NeuroFlow_newloss_GCN_other_firstGNN_noteBook_noline.py
```


**Task 2 — FlowMatching prior and image generation**

Option A — Run the notebook interactively (recommended for stepwise reproduction):

```
open Task2/Generation_metrics_sub8_GCN_notebook_woNSR.ipynb with Jupyter
```

Option B — Run training and inference scripts from command line:

```
cd Task2/
python newDiff_my_Notebook_fm.py        # train FlowMatching prior
python newDiff_my_gen_Notebook_fm.py    # inference: generate images from EEG embeddings
```

Adjust paths and hyperparameters inside the scripts .

## Visualization and Analysis
We provide several scripts and notebooks used for visualization in the paper:

- `GAM.py` — Global attention map visualizations and saliency analyses.
- `LinjieJZ.py` — Additional visualization utilities used in experiments.
- `Mne_miccai.ipynb` — Interactive MNE-based EEG visualizations and preprocessing steps.
- `ShowCovTwin.py` — Covariance / twin visual analyses.

## Baselines and Comparison Experiments
To reproduce the contrastive retrieval baselines and ablation studies, run:

```
python contrast_retrieval.py
python contrast_retrieval_newloss.py
```

Both scripts live in the `Task1/` folder and compare different loss variants and retrieval pipelines.

## Outputs and Example Artifacts
Trained models and logs are stored under `Task1/outputs/` organized by experiment name. Example subfolders include encoder checkpoints and generated samples (e.g., `BrainDreamerEEGEncoder_cliploss/`).

## Experimental Details (for paper reproducibility)
- Training details (batch size, optimizer, learning rate schedules) are defined inside each task script and configurable at the files.
- Seed and dataset splits: leave-one-subject-out configuration is implemented in `Task1/eegdatasets_leaveone.py` and `Task2/eegdatasets_leaveone.py`.








