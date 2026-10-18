import os
import hydra

import numpy as np
import pandas as pd
from rdkit.ML.Cluster import Butina
from skfp.distances import bulk_tanimoto_binary_distance
from skfp.fingerprints import ECFPFingerprint


def make_split(
    input_csv: str, 
    output_csv: str | None = None
):
    df = pd.read_csv(input_csv)
    if "fold" in df.columns:
        print('"fold" column already exists, skipping')
        return

    fp = ECFPFingerprint()
    fps = fp.transform(df["SMILES"])
    dists = bulk_tanimoto_binary_distance(fps)
    clusters = Butina.ClusterData(dists, len(dists), distThresh=0.35, isDistData=True)

    # greedy balanced assignment: place each cluster (largest first) into the
    # fold with the fewest members
    # dict: row index -> fold ID
    fold_sizes = [0, 0, 0, 0, 0]  # 5-fold CV
    folds = {}
    for cluster in sorted(clusters, key=len, reverse=True):
        f = int(np.argmin(fold_sizes))
        for idx in cluster:
            folds[idx] = f
        fold_sizes[f] += len(cluster)

    df["fold"] = folds

    output_path = output_csv if output_csv else input_csv
    df.to_csv(output_path, index=False)

@hydra.main(config_path="config", config_name="butina", version_base="1.3")
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
        output_csv=out_csv
    )

if __name__ == "__main__":
    main()
