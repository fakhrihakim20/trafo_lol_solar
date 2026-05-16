# IEEE ICT-PEP 2026 Paper Draft

**Title:** *Transformer Loss-of-Life Acceleration under High Rooftop-PV Penetration on Indonesian 150/20 kV Substations: A Hybrid IEEE C57.91 + XGBoost Simulation*

**Author:** Fakhri Hakim, PT PLN (Persero)
**Target venue:** IEEE International Conference on Technology and Policy in Electric Power and Energy (ICT-PEP) 2026
**Submission deadline:** 30 June 2026

---

## Files

| File | Purpose |
|---|---|
| `main.tex` | IEEEtran LaTeX source (6 pages, 2-column, 10 pt) |
| `refs.bib` | BibTeX file with 22 DOI-verified references |
| `main.md` | Markdown mirror with identical content (for review or pandoc → docx) |
| `figures/fig1_pypsa_network.{pdf,png}` | PyPSA two-bus network schematic (used in §II.C) |
| `figures/fig2_surrogate.{pdf,png}` | XGBoost surrogate accuracy (used in §IV.A) |
| `figures/fig3_pv_lol.{pdf,png}` | PV penetration vs LoL curves (used in §IV.B–C) |
| `figures/fig4_dispatch.{pdf,png}` | Threshold vs LP dispatch comparison (used in §IV.D) |

All figures are copies of the originals in `../results/figures/`. Re-running `python ../scripts/05_generate_figures.py` will refresh the source PDFs/PNGs; copy them back into `figures/` afterwards.

---

## Building the PDF

The standard IEEEtran build pipeline:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main      # second pass to resolve cross-references
```

This produces `main.pdf` (the camera-ready candidate). The four passes are required so that BibTeX picks up all `\cite{}` keys, the bibliography is written, and forward references stabilise.

### Requirements

A working LaTeX distribution with the following packages:

- `IEEEtran` (provides the conference document class)
- `amsmath`, `amssymb`, `amsfonts` (math)
- `graphicx` (figure inclusion)
- `booktabs` (publication-quality tables)
- `cite`, `url`, `xcolor`

On Windows: install MiKTeX (https://miktex.org) and let it auto-fetch missing packages on first compile.
On macOS: install MacTeX (https://www.tug.org/mactex/).
On Linux: `apt install texlive-publishers texlive-fonts-recommended texlive-latex-extra`.

### Quick Markdown preview

If you only want to read the prose without compiling LaTeX, open `main.md` in any Markdown viewer (VS Code, Typora, GitHub web preview). Equations render via MathJax in most viewers.

### Convert to Word

```bash
pandoc main.md -o main.docx --reference-doc=ieee_template.docx
```

(A `ieee_template.docx` is not bundled — apply IEEE Word template manually if a Word version is needed for co-author review.)

---

## Verifying the Numbers

Every quantitative claim in the paper traces back to artifacts in `../results/`:

| Claim | Source |
|---|---|
| Surrogate RMSE = 0.230 °C, R² = 0.9998 | `../results/metrics.json` keys `surrogate_rmse_c`, `surrogate_r2` |
| Baseline LoL = 0.0342 %/yr | `../results/tables/mc_results_all_scenarios.parquet`, scenario `baseline`, mean of `lol_percent_annual` |
| 75 % PV LoL = 0.0277 %/yr | same parquet, scenario `pv75_tropical_today` |
| 5 MWh BESS LoL = 0.0273 %/yr | same parquet, scenario `pv75_battery5mwh_today` |
| LP dispatch LoL = 0.02760 %/yr | `../results/tables/dispatch_comparison.csv`, `dispatch_method=pypsa_lp`, scenario `pv75_battery5mwh_today` |
| 41 000 MC runs in 53 min | `../results/metrics.json` keys `total_runs`, `runtime_seconds` (3211.9 s ≈ 53.5 min) |

To re-verify quickly:

```bash
python -c "
import json, pandas as pd
m = json.load(open('../results/metrics.json'))
print('RMSE:', m['surrogate_rmse_c'], 'R2:', m['surrogate_r2'])
mc = pd.read_parquet('../results/tables/mc_results_all_scenarios.parquet')
print(mc.groupby('scenario_name')['lol_percent_annual'].mean().round(4).head(10))
"
```

---

## Page-Length Tuning

If the compiled PDF runs over 6 pages:

1. Trim §II.D (XGBoost surrogate) to 1 paragraph; the methodology is well-established.
2. Move Table I to the column where the discussion of scenarios appears, and use `\footnotesize` for both tables.
3. Reduce Fig. 1 to single-column width (it is currently 0.9-column wide).
4. Cut the final paragraph of §V (transferability) — the conclusion repeats this point.

If under 6 pages, expand:

1. Add a §II.F sub-section on the moisture B-constant sensitivity (1 paragraph + 1 table row).
2. Expand the discussion of reverse-flow impact on LV-network voltage rise (½ paragraph in §V).

---

## Citation Notes

- **All 22 references in `refs.bib` are DOI- or URL-verified.** Do not add new citations without verifying via `https://doi.org/<DOI>`.
- Standards (refs [1]–[3]) are cited via the IEEE/IEC official catalogue URLs because they have no DOI in the journal sense.
- Reference [10] (Lesieutre 1997) is currently unused in the LaTeX prose; remove from `refs.bib` if BibTeX warns about it.

---

## Submission Checklist (post-acceptance)

When/if the paper is accepted at ICT-PEP 2026:

- [ ] Apply IEEE copyright form (separate process via the conference submission system)
- [ ] Replace email in `\IEEEauthorblockA{}` with the official corresponding-author address required by the venue
- [ ] Add a brief funding/acknowledgement statement if required
- [ ] Confirm all figures are vector PDF (already done)
- [ ] Re-compile with the conference's official IEEEtran style file if provided
- [ ] Generate the IEEE PDF eXpress validation report

---

*Generated 2026-04-27 from the validated `transformer_lol_sim` codebase (Phase 4D complete: 82/82 tests pass, surrogate RMSE 0.230 °C, 41 000-run MC sweep).*
