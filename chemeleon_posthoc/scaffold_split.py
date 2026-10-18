import os
import hydra

import numpy as np
import pandas as pd
from skfp.model_selection.splitters.scaffold_split import _create_scaffold_sets


def make_split(
    input_csv: str,
    output_csv: str | None = None,
    n_folds: int = 5,
    use_csk: bool = False,
):
    df = pd.read_csv(input_csv)
    if "fold" in df.columns:
        print('"fold" column already exists, skipping')
        return

    # groups of row indices sharing a Bemis-Murcko scaffold (same grouping as
    # skfp's scaffold_train_test_split)
    scaffold_sets = _create_scaffold_sets(df["SMILES"].tolist(), use_csk=use_csk)

    # greedy balanced assignment: place each scaffold set (largest first) into
    # the fold with the fewest members, so no scaffold spans two folds
    fold_sizes = [0] * n_folds
    folds = np.empty(len(df), dtype=int)
    for scaffold_set in sorted(scaffold_sets, key=len, reverse=True):
        f = int(np.argmin(fold_sizes))
        folds[scaffold_set] = f
        fold_sizes[f] += len(scaffold_set)

    print(f"{len(scaffold_sets)} scaffolds, largest: {max(map(len, scaffold_sets))}, fold sizes: {fold_sizes}")
    df["fold"] = folds

    output_path = output_csv if output_csv else input_csv
    df.to_csv(output_path, index=False)

@hydra.main(config_path="config", config_name="scaffold", version_base="1.3")
def main(cfg):
    in_dir = cfg.raw_data_dir
    out_dir = cfg.output_dir

    os.makedirs(out_dir, exist_ok=True)

    in_csv = os.path.join(in_dir, cfg.dataset.name + ".csv")
    out_csv = os.path.join(out_dir, cfg.dataset.name + "_split.csv")

    print(f"Input CSV: {in_csv}")
    print(f"Output CSV: {out_csv}")

    if os.path.exists(out_csv):
        print(f"Output CSV already exists: {out_csv}, skipping")
        return

    make_split(
        input_csv=in_csv,
        output_csv=out_csv,
        n_folds=cfg.n_folds,
        use_csk=cfg.use_csk,
    )

if __name__ == "__main__":
    main()
