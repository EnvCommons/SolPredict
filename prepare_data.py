"""
prepare_data.py - One-time data preparation for solpredict environment

Downloads AqSolDB and ESOL datasets, removes ESOL overlap from AqSolDB,
applies scaffold split, and saves training/validation/test files.

Usage:
    pip install rdkit tdc pandas requests
    python prepare_data.py
"""

import json
import pandas as pd
import requests
from pathlib import Path


def smiles_to_inchikey(smiles: str) -> str | None:
    """Convert SMILES to InChIKey for canonical matching."""
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey

        # Strip whitespace from SMILES
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return MolToInchiKey(mol)
    except Exception:
        pass
    return None


def download_esol() -> pd.DataFrame:
    """Download ESOL (Delaney) dataset from DeepChem's GitHub."""
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    print(f"  Downloading from {url}")

    response = requests.get(url)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    # The DeepChem ESOL has columns:
    # "Compound ID", "ESOL predicted log solubility in mols per litre",
    # "Minimum Degree", "Molecular Weight", "Number of H-Bond Donors",
    # "Number of Rings", "Number of Rotatable Bonds", "Polar Surface Area",
    # "measured log solubility in mols per litre", "smiles"

    # Rename to standard format and strip whitespace from SMILES
    df = df.rename(columns={
        "smiles": "SMILES",
        "measured log solubility in mols per litre": "LogS"
    })
    df["SMILES"] = df["SMILES"].str.strip()

    return df[["SMILES", "LogS"]]


def scaffold_split(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 42) -> tuple:
    """
    Split dataset by Bemis-Murcko scaffolds.
    Returns (train_df, val_df).
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import numpy as np

    np.random.seed(seed)

    # Generate scaffolds
    scaffolds = {}
    for idx, row in df.iterrows():
        smiles = row["SMILES"]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = smiles  # Fallback to original SMILES
        else:
            scaffold = smiles

        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (largest first for more stable splits)
    scaffold_sets = list(scaffolds.values())
    scaffold_sets.sort(key=len, reverse=True)

    # Assign scaffolds to train/val
    train_indices = []
    val_indices = []
    train_size = int(len(df) * train_frac)

    for scaffold_set in scaffold_sets:
        if len(train_indices) < train_size:
            train_indices.extend(scaffold_set)
        else:
            val_indices.extend(scaffold_set)

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)

    return train_df, val_df


def main():
    from tdc.single_pred import ADME

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Loading AqSolDB dataset...")
    aqsoldb = ADME(name="Solubility_AqSolDB").get_data()
    print(f"  Loaded {len(aqsoldb)} compounds from AqSolDB")

    print("Loading ESOL (Delaney) dataset...")
    esol = download_esol()
    print(f"  Loaded {len(esol)} compounds from ESOL")

    # Standardize column names for AqSolDB
    # TDC uses 'Drug' for SMILES and 'Y' for the target value
    aqsoldb = aqsoldb.rename(columns={"Drug": "SMILES", "Y": "LogS"})

    print("Generating InChIKeys for ESOL compounds...")
    esol["InChIKey"] = esol["SMILES"].apply(smiles_to_inchikey)
    esol_keys = set(esol["InChIKey"].dropna())
    print(f"  Generated {len(esol_keys)} unique InChIKeys for ESOL")

    print("Generating InChIKeys for AqSolDB compounds...")
    aqsoldb["InChIKey"] = aqsoldb["SMILES"].apply(smiles_to_inchikey)

    # Remove ESOL compounds from AqSolDB
    print("Removing ESOL overlap from AqSolDB...")
    aqsoldb_filtered = aqsoldb[~aqsoldb["InChIKey"].isin(esol_keys)].copy()
    removed_count = len(aqsoldb) - len(aqsoldb_filtered)
    print(f"  Removed {removed_count} overlapping compounds")
    print(f"  AqSolDB after filtering: {len(aqsoldb_filtered)} compounds")

    # Apply scaffold split (80/20 train/val)
    print("Applying scaffold split (80/20)...")
    train_df, val_df = scaffold_split(aqsoldb_filtered, train_frac=0.8, seed=42)

    print(f"  Train set: {len(train_df)} compounds")
    print(f"  Validation set: {len(val_df)} compounds")

    # Save training and validation data (agent sees these)
    train_df[["SMILES", "LogS"]].to_csv(data_dir / "train.csv", index=False)
    val_df[["SMILES", "LogS"]].to_csv(data_dir / "val.csv", index=False)
    print(f"  Saved train.csv and val.csv")

    # Save test SMILES only (agent sees this)
    esol[["SMILES"]].to_csv(data_dir / "test_smiles.csv", index=False)
    print(f"  Saved test_smiles.csv ({len(esol)} compounds)")

    # Save ground truth (hidden from agent, used for scoring)
    ground_truth = {row["SMILES"]: float(row["LogS"]) for _, row in esol.iterrows()}
    with open(data_dir / "test_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"  Saved test_ground_truth.json (hidden)")

    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Training compounds: {len(train_df)}")
    print(f"Validation compounds: {len(val_df)}")
    print(f"Test compounds (ESOL): {len(esol)}")
    print(f"\nLogS statistics:")
    print(f"  Train - mean: {train_df['LogS'].mean():.3f}, std: {train_df['LogS'].std():.3f}")
    print(f"  Val   - mean: {val_df['LogS'].mean():.3f}, std: {val_df['LogS'].std():.3f}")
    print(f"  Test  - mean: {esol['LogS'].mean():.3f}, std: {esol['LogS'].std():.3f}")

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
