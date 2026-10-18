"""LaTeX and figure-saving helpers shared by tables.py and figures.py."""

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger("published_figures")


def tex_escape(s: str) -> str:
    return s.replace("\\_", "_").replace("_", "\\_").replace("%", "\\%")


def pct(x: float) -> str:
    return f"{x:.2%}"


def write_tex(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n")
    log.info("Wrote %s", path)


def align_tex_rows(rows: list[str]) -> list[str]:
    """Pad cells with spaces so the `&` separators of the source line up across rows.

    Only rows with the common cell count are aligned (a \\multicolumn header row is left as is),
    and nested-tabular cells do not widen their column. Numbers are right-justified.
    """
    split = []
    for row in rows:
        m = re.match(r"(.*?)\s*(\\\\(?:\s*\\hline)?)\s*$", row, re.S)
        body, end = (m.group(1), m.group(2)) if m else (row, "")
        split.append(([c.strip() for c in body.split("&")], end))
    n_cells = pd.Series([len(cells) for cells, _ in split]).mode()[0]
    widths = [0] * n_cells
    for cells, _ in split:
        if len(cells) == n_cells:
            for i, c in enumerate(cells):
                if "\\begin{tabular}" not in c:
                    widths[i] = max(widths[i], len(c))

    def pad(c: str, i: int, cells: list[str]) -> str:
        if len(cells) != n_cells:
            return c
        if re.fullmatch(r"[+-]?[\d.]+(\\%)?", c):
            return c.rjust(widths[i])
        return c if i == n_cells - 1 else c.ljust(widths[i])  # no trailing spaces before \\

    return [" & ".join(pad(c, i, cells) for i, c in enumerate(cells)) + (f" {end}" if end else "")
            for cells, end in split]


def tex_table(env: str, colspec: str, header: str, rows: list[str], caption: str, label: str,
              placement: str = "") -> str:
    """Table in the manuscript's style: header, \\hline\\hline, rows, caption, label."""
    header_lines = header.split("\n")
    aligned = align_tex_rows(header_lines + rows)
    return "\n".join([
        f"\\begin{{{env}}}{placement}", "    \\centering", f"\\begin{{tabular}}{{{colspec}}}",
        *aligned[:len(header_lines)], "\\hline\\hline", *aligned[len(header_lines):],
        "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{env}}}",
    ])


def boxed_table(colspec: str, header: str, rows: list[str], caption: str, label: str, footer: list[str] = ()) -> str:
    """Table in the boxed style of the external-validation tables: |l|c|, \\hline after every row."""
    body = [f"{r}\n\\hline" for r in rows]
    if footer:
        body[-1] = body[-1].removesuffix("\\hline") + "\\hline\\hline"
        body += [f"{r}\n\\hline" for r in footer]
    return "\n".join(["\\begin{table}[htbp]", "\\centering", f"\\begin{{tabular}}{{{colspec}}}", "\\hline",
                      header, "\\hline", *body, "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      "\\end{table}"])


def save_png(fig, path: Path, dpi: int = 300, tight: bool = True, pad_inches: float = 0.02) -> None:
    """PNG only (what the manuscripts include); appends the extension, as names like rho_0.2 contain dots."""
    fig.savefig(path.parent / f"{path.name}.png", dpi=dpi, bbox_inches="tight" if tight else None,
                pad_inches=pad_inches)
    plt.close(fig)
    log.info("Wrote %s.png", path)


def compare_with_manuscript(tables_dir: Path, tex_files: list[Path]) -> list[str]:
    """Generated tables vs the tables of the manuscripts with the same label, ignoring whitespace."""
    src = "\n".join(p.read_text() for p in tex_files)
    manuscript = {}
    for env, _ in re.findall(r"(\\begin\{(table\*?)\}.*?\\end\{\2\})", src, re.S):
        if m := re.search(r"\\label\{([^}]*)\}", env):
            manuscript[m.group(1)] = env
    norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    lines = []
    for f in sorted(tables_dir.glob("*.tex")):
        label = re.search(r"\\label\{([^}]*)\}", f.read_text()).group(1)
        status = ("not in the manuscripts" if label not in manuscript
                  else "identical" if norm(f.read_text()) == norm(manuscript[label]) else "DIFFERS")
        lines.append(f"{label}: {status}")
    lines += [f"{label}: in the manuscripts, not generated" for label in sorted(manuscript)
              if not (tables_dir / f"{label.replace(':', '_')}.tex").exists()]
    return lines
