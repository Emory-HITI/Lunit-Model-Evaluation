# Subgroup Performance of a Commercial Digital Breast Tomosynthesis Model for Breast Cancer Detection

Code used for the evaluation of the Lunit INSIGHT DBT model on the Emory Breast Imaging Dataset (EMBED).

Preprint available at https://arxiv.org/abs/2503.13581

> Brown-Mulry, Beatrice, Rohan Satya Isaac, Sang Hyup Lee, Ambika Seth, KyungJee Min, Theo Dapamede, Frank Li et al. "Subgroup Performance of a Commercial Digital Breast Tomosynthesis Model for Breast Cancer Detection." arXiv preprint arXiv:2503.13581 (2025).

## Installation

Analysis was conducted on Python 3.11.10 running on JupyterLab v4.0.11 on Ubuntu 22.04.5 LTS

Python packages used for the analysis are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`

## Usage

### Data Engineering

`notebooks/screening_label_assignment.ipynb`

This file contains the label assignment pipeline to assign ground truth labels to the raw EMBED dataset.

### Analysis

`notebooks/model_performance_analyses.ipynb`

This file contains the model evaluation code which runs on the data produced by `screening_label_assignment.ipynb`.

`notebooks/*.py`

These files contain some support functions and constants used by `model_performance_analyses.ipynb`.
