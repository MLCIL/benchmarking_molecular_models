"""Constants of the JCIM revision (main.tex + supplementary.tex): inputs, names, protocol and test settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # published_figures/
DEFAULT_RESULTS = ROOT.parent / "results" / "jcim_revision_2026_10"
DEFAULT_OUT = ROOT / "output"
STATIC = ROOT / "static"

# Files of the results directory
RESULTS_DB = "all.db"
POSTHOC_CSV = "posthoc_scaffold.csv"
RUNTIME_CSV = "runtime.csv"
BENCHMARK_DATASETS_CSV = "benchmark_datasets.csv"
EXTERNAL_DATASETS_DIR = "external_datasets"

# Output layout = the paths main.tex / supplementary.tex include
FIGURES = "figures/updated"
TABLES = "tables"

SEED = 42
A4_TEXT_WIDTH = 6.9  # inches, full text width of an A4 page with ~2 cm margins

# --------------------------------------------------------------------------- benchmark

TIE = 0.01  # AUROC difference treated as a tie / near-win
BASELINE = "ECFP_count"
HEADS = {"knn": "knn", "rf": "rf", "ridge": "linear"}  # DB name -> name in the tables

# 18 TDC + 7 MoleculeNet datasets
DATASETS = [
    "AMES", "Bioavailability_Ma", "CYP1A2_Veith", "CYP2C19_Veith",
    "CYP2C9_Substrate_CarbonMangels", "CYP2C9_Veith", "CYP2D6_Substrate_CarbonMangels",
    "CYP2D6_Veith", "CYP3A4_Substrate_CarbonMangels", "CYP3A4_Veith", "DILI", "HIA_Hou",
    "PAMPA_NCATS", "Pgp_Broccatelli", "SARSCoV2_3CLPro_Diamond", "SARSCoV2_Vitro_Touret",
    "hERG", "hERG_Karim", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molhiv",
    "ogbg-molmuv", "ogbg-molsider", "ogbg-moltox21",
]
DATASET_NAMES = {
    "ogbg-molclintox": "MoleculeNet_ClinTox",
    "ogbg-molmuv": "MoleculeNet_MUV",
    "ogbg-molbbbp": "MoleculeNet_BBBP",
    "ogbg-molhiv": "MoleculeNet_HIV",
    "ogbg-molbace": "MoleculeNet_BACE",
    "ogbg-moltox21": "MoleculeNet_Tox21",
    "ogbg-molsider": "MoleculeNet_SIDER",
}

# The 53 variants of the 28 models: embedder -> (model name, variant name)
MODELS = {
    "AtomPair": ("Atom Pair", "Atom Pair"),
    "AtomPair_count": ("Atom Pair", "Atom Pair [Count]"),
    "CDDD": ("CDDD", "CDDD"),
    "ChemBERTa-5M-MLM": ("ChemBERTa", "ChemBERTa [5M][MLM]"),
    "ChemBERTa-5M-MTR": ("ChemBERTa", "ChemBERTa [5M][MTR]"),
    "ChemBERTa-10M-MLM": ("ChemBERTa", "ChemBERTa [10M][MLM]"),
    "ChemBERTa-10M-MTR": ("ChemBERTa", "ChemBERTa [10M][MTR]"),
    "ChemBERTa-77M-MLM": ("ChemBERTa", "ChemBERTa [77M][MLM]"),
    "ChemBERTa-77M-MTR": ("ChemBERTa", "ChemBERTa [77M][MTR]"),
    "chemeleon": ("CheMeleon", "CheMeleon"),
    "ChemFM-1B": ("ChemFM", "ChemFM [1B]"),
    "ChemFM-3B": ("ChemFM", "ChemFM [3B]"),
    "chemformer_augment": ("Chemformer", "Chemformer [Augment]"),
    "chemformer_mask": ("Chemformer", "Chemformer [Mask]"),
    "chemformer_combined": ("Chemformer", "Chemformer [Combined]"),
    "ChemGPT-4.7M": ("ChemGPT", "ChemGPT [4.7M]"),
    "ChemGPT-19M": ("ChemGPT", "ChemGPT [19M]"),
    # The leak-free retraining (Hugging Face scikit-fingerprints/benchmarking_models_clamp_refitted). The DB's `CLAMP`
    # holds the authors' weights, which saw benchmark molecules during pretraining; it is not used.
    "CLAMP_refitted": ("CLAMP", "CLAMP"),
    "coati": ("COATI", "COATI"),
    "GNN-CP-sum": ("GIN", "ContextPred"),  # GIN + context prediction (GraphMVP checkpoint), sum pooling
    "ECFP": ("ECFP", "ECFP"),
    "ECFP_count": ("ECFP", "ECFP [Count]"),
    "GEM": ("GEM", "GEM"),
    "GraphFP-C": ("GraphFP", "GraphFP [C]"),
    "GraphFP-CP": ("GraphFP", "GraphFP [CP]"),
    "GraphMVP_AM-max": ("GraphMVP", "GraphMVP [AM][Max]"),
    "GraphMVP_AM-mean": ("GraphMVP", "GraphMVP [AM][Mean]"),
    "GraphMVP_AM-sum": ("GraphMVP", "GraphMVP [AM][Sum]"),
    "GraphMVP_CP-max": ("GraphMVP", "GraphMVP [CP][Max]"),
    "GraphMVP_CP-mean": ("GraphMVP", "GraphMVP [CP][Mean]"),
    "GraphMVP_CP-sum": ("GraphMVP", "GraphMVP [CP][Sum]"),
    "grover_base": ("GROVER", "GROVER [Base]"),
    "grover_large": ("GROVER", "GROVER [Large]"),
    "mat_masking_200k": ("MAT", "MAT [200k]"),
    "mat_masking_2M": ("MAT", "MAT [2M]"),
    "mat_masking_20M": ("MAT", "MAT [20M]"),
    "mol2vec": ("Mol2Vec", "Mol2Vec"),
    "molbert": ("MolBERT", "MolBERT"),
    "mole": ("MolE", "MolE"),
    "MoLFormer-XL-both-10pct": ("MoLFormer", "MoLFormer"),
    "mol_r_gat_1024": ("MolR", "MolR [GAT][1024]"),
    "mol_r_gcn_1024": ("MolR", "MolR [GCN][1024]"),
    "mol_r_sage_1024": ("MolR", "MolR [SAGE][1024]"),
    "mol_r_tag_1024": ("MolR", "MolR [TAG][1024]"),
    "mordred_2d": ("Mordred", "Mordred [2D]"),
    "rmat_4M": ("R-MAT", "R-MAT [4M]"),
    "SELFormer": ("SELFormer", "SELFormer"),
    "SELFormer-Lite": ("SELFormer", "SELFormer [Lite]"),
    "SimSon": ("SimSon", "SimSon"),
    "TT": ("TT", "TT"),
    "TT_count": ("TT", "TT [Count]"),
    "unimolv1": ("Uni-Mol", "Uni-Mol"),
    "unimolv2": ("Uni-Mol2", "Uni-Mol2"),
}
# Variant whose row of runtime.csv times an embedder, where it differs: CLAMP was timed with the authors' checkpoint,
# which has the same architecture as the retrained model
RUNTIME_EMBEDDERS = {"CLAMP_refitted": "CLAMP"}
# Fingerprints always use their count variant (ECFP count is the baseline); GIN is ContextPred (its only variant)
FIXED_REPRESENTATIVES = {"ECFP": BASELINE, "Atom Pair": "AtomPair_count", "TT": "TT_count", "GIN": "GNN-CP-sum"}
# Name of a model in the model-level tables and figures when this variant represents it
REP_LABELS = {"GNN-CP-sum": "ContextPred"}
FINGERPRINTS = ["ECFP", "Atom Pair", "TT", "Mordred"]  # the 4 models that are not pretrained neural networks
# The main text spells Mol2vec as its authors do; the supplement and the figures use Mol2Vec
MAIN_TEXT_NAMES = {"Mol2Vec": "Mol2vec"}

# --------------------------------------------------------------------------- Bayesian Bradley-Terry

BBT_ROPE = (0.35, 0.65)
BBT_ROPES = [(0.45, 0.55), (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8)]
# Refit on the 28 representatives; the paper's 2000 / 1000 does not converge (R-hat 1.03) with 28 models
BBT_SAMPLING = dict(draws=10000, tune=2000, target_accept=0.95)
# Fit on all 53 variants (selects the representatives); mixes more slowly
BBT_VARIANT_SAMPLING = dict(draws=40000, tune=10000, target_accept=0.95)
BBT_DECISIONS = ["Better", "Practically equivalent", "Unknown", "Worse"]
# Strong interpretation, ordered along P(model > ECFP); "Unknown" split by the side of the equivalence region
STRONG_DECISIONS = ["Better", "Unknown (equivalent/better)", "Practically equivalent",
                    "Unknown (worse/equivalent)", "Worse"]

# --------------------------------------------------------------------------- external validation (t-test)

POSTHOC_MODELS = {"chemeleon": "CheMeleon", "ecfp_count": "ECFP"}  # left vs right model of the t-test
# File id -> name in the manuscript, in the order of tab:external_datasets
POSTHOC_DATASETS = {
    "NICEATM_toxicity": "NICEATM Acute Oral Toxicity",
    "rat_carcinogenicity": "Rat carcinogenicity",
    "DILIrank_2_0": "DILIrank 2.0",
    "B3DB_classification": "B3DB",
    "BSEP": "BSEP",
    "mitochondrial_toxicity": "Mitochondrial toxicity",
    "skin_sensitization": "Skin sensitization",
    "bioavailability": "Bioavailability",
    "eye_corrosion": "Eye corrosion",
    "eye_irritation": "Eye irritation",
    "human_photosensitizers": "Human photosensitizers",
}
POSTHOC_DATASET_CITES = {
    "NICEATM_toxicity": "dataset_external_NICEATM,dataset_external_NICEATM_2",
    "rat_carcinogenicity": "dataset_external_rat_carcinogenicity",
    "DILIrank_2_0": "dataset_external_dilirank_2",
    "B3DB_classification": "dataset_external_B3DB",
    "BSEP": "dataset_external_bsep",
    "mitochondrial_toxicity": "dataset_external_mitochondrial_toxicity",
    "skin_sensitization": "dataset_external_skin_toxicity",
    "bioavailability": "dataset_external_bioavailability",
    "eye_corrosion": "dataset_external_eye_irritation_corrosion",
    "eye_irritation": "dataset_external_eye_irritation_corrosion",
    "human_photosensitizers": "dataset_external_photosensitizers",
}
# Endpoint column of tab:external_datasets (descriptive, not computed)
POSTHOC_DATASET_ENDPOINTS = {
    "NICEATM_toxicity": "Acute oral rat toxicity, non-toxic (0) or very toxic (1)",
    "rat_carcinogenicity": "In vivo classification as carcinogen or non-carcinogen",
    "DILIrank_2_0": "Drug-induced liver injury: none (0), low (1), high (2)",
    "B3DB_classification": "Blood-brain barrier penetration, $\\log BB$ threshold $-1$",
    "BSEP": "Bile salt export pump inhibition, $\\mathrm{pIC}_{50} = 4$ threshold",
    "mitochondrial_toxicity": "Toxicity to mitochondria",
    "skin_sensitization": "Induction of allergic contact dermatitis",
    "bioavailability": "Human oral bioavailability $F \\geq 50\\%$",
    "eye_corrosion": "Eye corrosion per GHS H314/H318 classification",
    "eye_irritation": "Eye irritation per GHS H319 classification",
    "human_photosensitizers": "Induction of photosensitization in humans",
}
# Two-line names of the supplement's statistics table
POSTHOC_DATASET_TEX_NAMES = {"NICEATM_toxicity": "NICEATM Acute \\\\ Oral Toxicity",
                             "human_photosensitizers": "Human \\\\ photosensitizers"}
SPLIT = "scaffold"
# Label -> rho. None -> Nadeau-Bengio rho = 1/k from the folds (every fold holds 20% of the data on average).
# "0" is fitted as 1e-9: the library requires 0 < rho < 1; it treats folds as independent.
POSTHOC_RHOS = {"0": 1e-9, "0.001": 0.001, "0.01": 0.01, "0.05": 0.05, "0.1": 0.1, "0.2": None}
POSTHOC_MAIN_RHO = "0.2"
POSTHOC_FOREST_RHO = "0"  # Figure "posthoc_forest": independent folds, the least conservative setting
POSTHOC_SIMPLEX_RHOS = ("0", "0.2")
# The centered form diverges on this data; the non-centered form is the same model
POSTHOC_SAMPLING = dict(draws=5000, tune=2000, chains=4, target_accept=0.95)
NU_PRIORS = {"corani": "Main analysis", "juarez_steel": "Ju\\'arez \\& Steel"}  # prior on nu -> table name
JS_SHAPE, JS_RATE = 2.0, 0.1  # Juarez & Steel (2010): nu ~ Gamma(2, 0.1), shape and rate
