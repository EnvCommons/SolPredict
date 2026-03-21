# SolPredict

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SolPredict)

## Description

SolPredict is an ORS environment for evaluating an agent's ability to develop ML models that predict aqueous solubility (LogS) from molecular SMILES notation. Agents are given training data from the AqSolDB database, develop and train predictive models in a sandboxed compute environment, and submit predictions for test molecules from the ESOL (Delaney) dataset.

## Capabilities

- Developing machine learning models for molecular property prediction
- Feature engineering from SMILES molecular representations
- Working with cheminformatics libraries (RDKit, scikit-learn, etc.)
- Multi-step model development and evaluation

## Compute Requirements

Agents are given a sandbox with 4 CPUs and 8GB RAM, with network access enabled for installing additional packages.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There is one split in this environment:

- **Train**: 1 task (`solubility_train`)

The single task involves predicting aqueous solubility for 1,128 test molecules.

## Reward Structure

This is a multi-turn, sandbox-based environment. The agent develops a model, generates predictions as a CSV file, and submits via the `submit` tool. The reward is normalized against a naive baseline (predicting the training mean): `reward = 1 - RMSE / baseline_RMSE`. Positive reward means the agent outperformed the baseline; 1.0 would be perfect predictions. Scoring is deterministic -- no LLM graders are used.

## Data

Three CSV files are provided to agents:

- `train.csv`: 7,093 compounds with SMILES and LogS values (from AqSolDB, scaffold-split, ESOL removed)
- `val.csv`: 1,772 compounds for validation (80/20 split from AqSolDB)
- `test_smiles.csv`: 1,128 test molecules (SMILES only, from ESOL/Delaney dataset)

Hidden ground truth contains LogS values for the 1,123 matched test compounds.

## Tools

Agents get CLI tools (`bash`, `read`, `write`, `edit`, `multi_edit`, `grep`, `glob`, `ls`, `todo_write`) plus 1 environment-specific tool:

| Tool | Description |
|------|-------------|
| `submit` | Submit a CSV file with SMILES and LogS predictions for RMSE evaluation. |

## Time Horizon

SolPredict is an open-ended, multi-turn environment. Agents explore data, develop models, iterate on feature engineering and hyperparameter tuning, and submit predictions.

## Environment Difficulty

The environment tests end-to-end ML development skills including data exploration, feature engineering, model selection, and evaluation. Agents must work with molecular representations (SMILES) which require domain-specific processing (e.g., molecular fingerprints via RDKit).

## Other Environment Requirements

There are no further environment requirements; SolPredict works out of the box with the OpenReward endpoint without any external API keys.

## Safety

Agents in SolPredict develop molecular property prediction models in a sandboxed environment. There is a dual-use concern in that improved molecular prediction capabilities could be applied to both beneficial and harmful purposes.

## Citations

```bibtex
@article{sorkun2019aqsoldb,
  author    = {Murat Cihan Sorkun and Abhishek Khetan and S\"{u}leyman Er},
  title     = {AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds},
  journal   = {Scientific Data},
  volume    = {6},
  pages     = {143},
  year      = {2019},
  doi       = {10.1038/s41597-019-0151-1},
  url       = {https://www.nature.com/articles/s41597-019-0151-1}
}
```
