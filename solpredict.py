"""
Solpredict - Aqueous Solubility Prediction Environment

Multi-step CLI environment where agents train ML models to predict
aqueous solubility (LogS) from molecular SMILES notation.

Training: AqSolDB (scaffold-split, ESOL removed)
Test: ESOL (external validation set)
Reward: 1 - RMSE/baseline_RMSE (normalized against naive mean predictor)
"""

import json
import numpy as np
import os
import shlex
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from pydantic import BaseModel

from cli_environment import CLIEnvironment
from openreward.environments import tool, JSONObject, ToolOutput, TextBlock
from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings

from utils import download_text


# Path handling for data files
if os.path.exists("/orwd_data"):
    DATA_PATH = Path("/orwd_data/")
else:
    DATA_PATH = Path(__file__).parent


def load_ground_truth() -> dict[str, float]:
    """Load hidden test ground truth at module import time."""
    gt_file = DATA_PATH / "test_ground_truth.json"
    if gt_file.exists():
        with open(gt_file, "r") as f:
            return json.load(f)
    return {}


GROUND_TRUTH = load_ground_truth()
if GROUND_TRUTH:
    print(f"Loaded ground truth for {len(GROUND_TRUTH)} test compounds")


# Baseline RMSE: naive predictor (training mean = -2.7144 for all test compounds)
BASELINE_RMSE = 2.1203


class SubmissionUnavailable(Exception):
    """The agent did not write a submission file at the path it named."""


class TaskSpec(BaseModel):
    id: str


class SubmitParams(BaseModel, extra="forbid"):
    """Submit predictions for the test set."""
    submission_path: str


class SolpredictEnvironment(CLIEnvironment):
    """
    Solubility prediction environment.

    Agent has access to:
    - /data/train.csv: Training data with SMILES and LogS
    - /data/val.csv: Validation data for model tuning
    - /data/test_smiles.csv: Test SMILES (no LogS - agent must predict)

    Agent submits predictions via submit() tool.
    Reward is -RMSE on the hidden test set.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec, secrets=secrets)

        self.validated = TaskSpec.model_validate(task_spec)

        # Validate required API key
        api_key = secrets.get("api_key")
        if not api_key:
            raise ValueError("OpenReward API key must be provided via secrets parameter")

        # Set up sandbox with data files mounted
        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/SolPredict",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="4:8",  # 4 CPU, 8GB RAM
            block_network=False,  # Allow pip install
            bucket_config=SandboxBucketConfig(
                mount_path="/orwd_data",
                read_only=True,
                only_dir="data",
            )
        )

        or_client = AsyncOpenReward(api_key=api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        # Load ground truth for scoring
        self.ground_truth = GROUND_TRUTH

        self.todos: List[Dict[str, Any]] = []

    async def get_prompt(self) -> List[TextBlock]:
        """Return the task prompt for the agent."""
        prompt = """You are a molecular property prediction expert.

## Task
Predict aqueous solubility (LogS) for molecules in the test set.

## Available Data Files (mounted at /orwd_data/)
- /orwd_data/train.csv: Training data with columns [SMILES, LogS]
- /orwd_data/val.csv: Validation data with columns [SMILES, LogS]
- /orwd_data/test_smiles.csv: Test molecules with column [SMILES] (you must predict LogS)

## Submission Format
Create a CSV file with your predictions:
- Columns: SMILES, LogS
- One row per test molecule
- LogS values should be floating point numbers

Example submission.csv:
```
SMILES,LogS
CCO,-0.77
c1ccccc1,-2.18
```

## Available Tools
- Full CLI access: bash, read, write, edit, glob, grep, ls
- You may install packages with pip (e.g., rdkit, scikit-learn, xgboost, pytorch)
- For RDKit use `pip install rdkit`. The `rdkit-pypi` package is deprecated and is not
  binary-compatible with the numpy already installed here, so it will segfault on import.
- `bash` takes an optional `timeout` in seconds (default 300, maximum 1800). Long training
  runs need it raised explicitly.
- When ready, use submit(submission_path="/path/to/submission.csv") to submit predictions. It's recommend you put it in /home/ubuntu/submission.csv.

## Scoring
Your predictions will be scored using RMSE (Root Mean Squared Error) against the true LogS values.
Lower RMSE is better. Reward = 1 - RMSE/baseline, where baseline is the naive mean predictor.
Positive reward means you beat the baseline; 1.0 would be perfect.

Good luck!
"""
        return [TextBlock(text=prompt)]

    async def _read_submission(self, path: str) -> str:
        _, code = await self.sandbox.run(f"test -f {shlex.quote(path)}")
        if code != 0:
            raise SubmissionUnavailable(f"{path} does not exist in the sandbox")
        return await download_text(self.sandbox, path)

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit predictions for scoring against the hidden test set."""
        if not self.ground_truth:
            raise RuntimeError(
                f"No test ground truth available at {DATA_PATH}; refusing to score this submission"
            )

        try:
            content = await self._read_submission(params.submission_path)
        except SubmissionUnavailable as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error: {e}. Write your predictions there first.")],
                reward=0.0,
                finished=False
            )

        try:
            submission_df = pd.read_csv(StringIO(content))
        except Exception as e:
            return ToolOutput(
                metadata={"error": f"could not parse {params.submission_path} as CSV: {e}"},
                blocks=[TextBlock(text=f"Error: could not parse {params.submission_path} as CSV: {e}")],
                reward=0.0,
                finished=False
            )

        # Validate columns
        if "SMILES" not in submission_df.columns or "LogS" not in submission_df.columns:
            return ToolOutput(
                metadata={"error": "Submission must have columns: SMILES, LogS"},
                blocks=[TextBlock(text="Error: Submission must have columns: SMILES, LogS")],
                reward=0.0,
                finished=False
            )

        # Build predictions dict
        predictions = {}
        for _, row in submission_df.iterrows():
            smiles = str(row["SMILES"]).strip()
            try:
                logs = float(row["LogS"])
                predictions[smiles] = logs
            except (ValueError, TypeError):
                pass

        # Match predictions to ground truth
        y_true = []
        y_pred = []
        missing = []

        for smiles, true_logs in self.ground_truth.items():
            if smiles in predictions:
                y_true.append(true_logs)
                y_pred.append(predictions[smiles])
            else:
                missing.append(smiles)

        if len(y_true) == 0:
            return ToolOutput(
                metadata={"error": "No matching SMILES found in submission"},
                blocks=[TextBlock(text="Error: No matching SMILES found. Check your SMILES format.")],
                reward=0.0,
                finished=False
            )

        # Calculate RMSE
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # Normalized reward: 1.0 = perfect, 0.0 = baseline (train mean), negative = worse
        reward = 1.0 - (rmse / BASELINE_RMSE)

        # Build result message
        coverage = len(y_true) / len(self.ground_truth) * 100

        result_parts = [
            "## Submission Results",
            "",
            f"Predictions submitted: {len(predictions)}",
            f"Predictions matched: {len(y_true)} / {len(self.ground_truth)} ({coverage:.1f}% coverage)",
            "",
            "### Scoring",
            f"RMSE: {rmse:.4f}",
            f"Baseline RMSE: {BASELINE_RMSE:.4f}",
            f"Reward: {reward:.4f} (>0 = better than baseline)",
        ]

        if missing:
            result_parts.extend([
                "",
                "### Missing Predictions",
                f"First 10 missing: {missing[:10]}"
            ])

        result = "\n".join(result_parts)

        return ToolOutput(
            metadata={
                "task_id": self.validated.id,
                "predictions_submitted": len(predictions),
                "predictions_matched": len(y_true),
                "coverage": coverage,
                "rmse": rmse,
                "baseline_rmse": BASELINE_RMSE,
                "reward": reward,
            },
            blocks=[TextBlock(text=result)],
            reward=reward,
            finished=True
        )

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return single task per split (one prediction task)."""
        if split not in ["train"]:
            return []
        return [{"id": f"solubility_{split}"}]
