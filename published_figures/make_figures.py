"""Reproduce every figure, table and in-text number of the JCIM revision of the paper (arXiv:2508.06199).

Reads only the results directory (default results/jcim_revision_2026_10/) and writes, under --out, the paths
main.tex and supplementary.tex include:

    figures/updated/*.png   figures (src/figures.py)
    figures/workflow.png    static drawing (published_figures/static/)
    tables/*.tex, *.csv     tables, named after their labels (src/tables.py)
    numbers.md, numbers.csv numbers quoted in the text, checked against the article (src/numbers.py)
    cache/                  fitted hierarchical t-tests, reused while the fold scores and settings are unchanged

Run from the repo root (~10 min; the t-test fits use --jobs x --cores cores). The vendored, modified bayes-testkit
is an editable overlay, the root environment is untouched:

    ~/.local/bin/uv run --no-sync --with-editable published_figures/bayes-testkit \\
        python published_figures/make_figures.py [--results DIR] [--out DIR]
"""

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

from src import config as C
from src import figures, numbers, tables
from src.analysis import Analysis
from src.latex import compare_with_manuscript

log = logging.getLogger("published_figures")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=C.DEFAULT_RESULTS, help="results directory (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=C.DEFAULT_OUT, help="output directory (default: %(default)s)")
    ap.add_argument("--only", nargs="+", choices=["tables", "figures", "numbers"],
                    default=["tables", "figures", "numbers"], help="object types to generate (default: all)")
    ap.add_argument("--cores", type=int, default=4, help="cores per MCMC fit (4 chains)")
    ap.add_argument("--jobs", type=int, default=12, help="hierarchical t-test fits run in parallel")
    ap.add_argument("--refit", action="store_true", help="refit the cached hierarchical t-tests")
    ap.add_argument("--check-tex", nargs="+", type=Path, metavar="TEX",
                    help="main.tex / supplementary.tex: compare the generated tables with theirs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ["pymc", "arviz", "matplotlib.font_manager"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)

    out = args.out
    (out / C.FIGURES).mkdir(parents=True, exist_ok=True)
    (out / C.TABLES).mkdir(parents=True, exist_ok=True)
    a = Analysis(args.results, out / "cache", cores=args.cores, jobs=args.jobs, refit=args.refit)

    if "tables" in args.only:
        for make in tables.MAIN + tables.SUPPLEMENT:
            make(a, out)
    if "figures" in args.only:
        for make in figures.MAIN + figures.SUPPLEMENT:
            make(a, out)
    if "numbers" in args.only:
        results = {part: [(make.__doc__.splitlines()[0], make(a)) for make in fns]
                   for part, fns in [("Main text", numbers.MAIN), ("Supplement", numbers.SUPPLEMENT)]}
        (out / "numbers.md").write_text(numbers.to_markdown(results) + "\n")
        pd.DataFrame([{"part": part, "section": doc, **vars(n)} for part, groups in results.items()
                      for doc, ns in groups for n in ns]).to_csv(out / "numbers.csv", index=False)
        bad = [n for groups in results.values() for _, ns in groups for n in ns if n.ok is False]
        log.info("Wrote %s: %d numbers differ from the article", out / "numbers.md", len(bad))
        for n in bad:
            log.warning("Differs from the article: %s: article %s, computed %s", n.statement, n.article, n.computed)
    if args.check_tex:
        for line in compare_with_manuscript(out / C.TABLES, args.check_tex):
            log.info("Table %s", line)


if __name__ == "__main__":
    main()
