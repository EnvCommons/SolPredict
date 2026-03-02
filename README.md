# SolPredict

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SolPredict)

## Description

**SolPredict** is a multi-step CLI environment where agents develop machine learning models to predict aqueous solubility (LogS) from molecular SMILES notation. Agents have access to a sandboxed compute environment with training data ([AqSolDB](https://www.nature.com/articles/s41597-019-0151-1), scaffold-split with [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x) removed) and must build, train, and evaluate a predictive model, then submit predictions on a held-out test set (ESOL).

## Capabilities

- Developing machine learning models for molecular property prediction
- Feature engineering from SMILES notation (e.g., molecular descriptors, fingerprints)
- Training and evaluating regression models for aqueous solubility
- Multi-step iterative model development with CLI tools

## Compute Requirements

Agents in SolPredict are given a sandbox with 4 CPUs and 8GB RAM, network access enabled, and a Python 3.12 data science image.

## License

[MIT](https://opensource.org/license/mit).

## Tasks

There is a single task in the train split. This task requires the agent to:

1. Explore the provided training data (`train.csv` with SMILES and LogS columns, 7,093 compounds) and validation data (`val.csv`, 1,772 compounds).
2. Develop a predictive model for aqueous solubility (LogS).
3. Generate predictions for the test molecules (`test_smiles.csv`, 1,123 unique compounds, SMILES only).
4. Submit predictions as a CSV file with SMILES and LogS columns via the `submit` tool.

Training data is sourced from [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) (scaffold-split, with [ESOL](https://pubs.acs.org/doi/10.1021/ci034243x) molecules removed). The test set consists of ESOL molecules used as an external validation set.

## Reward Structure

This is a sparse, verifiable reward environment. The agent submits predictions once, and the environment scores them against hidden ground truth values. The reward is the negative Root Mean Squared Error (RMSE):

$$\text{Reward} = -\text{RMSE} = -\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2}$$

Higher (less negative) reward is better, with a maximum of 0.0 for perfect predictions.

Note that this is unnormalised, and we rely on group advantage normalisation during training to achieve this; but if you need normalisation *ex ante*, it is recommended to take the returned reward from the tool result and normalise it.

We do not use LLM graders for this task.

## Data

Agents are provided with three data files mounted in the sandbox:

- `train.csv`: Training data with columns [SMILES, LogS] from [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) (scaffold-split, ESOL removed) — 7,093 compounds
- `val.csv`: Validation data with columns [SMILES, LogS] — 1,772 compounds
- `test_smiles.csv`: Test molecules with column [SMILES] (no LogS — agent must predict) — 1,123 unique compounds

Data files are stored on the OpenReward platform. Training data is sourced from [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1); test data is sourced from the [ESOL (Delaney)](https://pubs.acs.org/doi/10.1021/ci034243x) dataset.

## Tools

Agents are given access to CLI tools for creating, viewing, and searching a filesystem (bash, glob, grep, ls, read, write, edit, multi_edit, todo_write). They are also given one environment-specific tool:

- `submit`: Submit a CSV file of predictions (columns: SMILES, LogS) for scoring against the hidden test set. Returns RMSE, coverage, and reward.

## Time Horizon

SolPredict is an open-ended, multi-step environment. Agents iteratively develop models using CLI tools before submitting final predictions. The number of tool calls varies depending on the agent's approach to model development.

## Other Environment Requirements

There are no further environment requirements beyond the OpenReward platform.

## Safety

Agents in SolPredict are asked to predict aqueous solubility values for molecules. The environment does not present direct safety risks, as agents only interact with molecular data within a sandboxed environment. No real-world chemical experiments are involved.

However, this domain is dual-use, meaning training a model to be good at this task could increase capabilities that could be misused when combined in other agentic workflows.

## Citations

```bibtex
@dataset{GRSolPredict,
  author    = {General Reasoning Inc. Team},
  title     = {SolPredict},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/GRSolPredict}
}
```

```bibtex
@article{sorkun2019aqsoldb,
  title={AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds},
  author={Sorkun, Murat Cihan and Khetan, Abhishek and Er, S{\"u}leyman},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={143},
  year={2019},
  publisher={Nature Publishing Group}
}

@article{delaney2004esol,
  title={ESOL: Estimating aqueous solubility directly from molecular structure},
  author={Delaney, John S},
  journal={Journal of Chemical Information and Computer Sciences},
  volume={44},
  number={3},
  pages={1000--1005},
  year={2004},
  publisher={ACS Publications}
}
```
