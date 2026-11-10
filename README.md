# Medicinal Chemistry Data and AI-Evaluation Portfolio

[![Tests](https://github.com/yazanus94-dot/Medchem-AI-Data-Science-/actions/workflows/ci.yml/badge.svg)](https://github.com/yazanus94-dot/Medchem-AI-Data-Science-/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Three self-contained Python projects that mirror the day-to-day work of a bioinformatics / medicinal-chemistry
data scientist who helps train and assess AI systems: **curating bioactivity data, modelling structure-activity
relationships, and grading the scientific accuracy of AI-generated answers.**

Author: Yazan Aljabari (biomedical engineer; Python scientific computing, data validation, technical reporting)

| # | Project | Question it answers | Headline result |
|---|---|---|---|
| 1 | [Bioactivity data QC and curation](project-1-bioactivity-data-qc/) | Can we trust the potency numbers in a ChEMBL-style export? | Curation cut potency error (RMSE) from 0.68 to 0.41 pIC50 units against known truth; 94% of injected problems caught |
| 2 | [SAR modelling with honest validation](project-2-sar-modeling/) | How well do models predict activity, and for which compounds do they fail? | Error rises from 0.38 to 0.46 RMSE when a whole amide-substituent group is held out; ridge on fragments degrades most (0.39 to 0.68) |
| 3 | [Scientific-answer evaluation harness](project-3-llm-science-eval/) | How do we grade AI answers to medicinal-chemistry and omics questions, and can we trust the grader? | Rubric scorer agrees with the answer key on 99.7% of 288 responses (Cohen's kappa 0.99) |

All three map directly onto the responsibilities of the role: analyzing medicinal-chemistry datasets,
curating and validating them, interpreting experimental data, assessing AI outputs for scientific accuracy,
and writing actionable feedback.

## Important: what the data is

**All data is synthetic, and that is deliberate.** Every dataset is generated from an explicit, documented
ground-truth rule (a 400-compound benzamide library with a known SAR for a hypothetical target). Because the truth is
known, every claim in this repository is *measured* (did the QC pipeline find the injected error? did the model
recover the real effect?) instead of assumed. The schemas mirror real ChEMBL exports, so the pipelines can be pointed
at real data. **No result here is a statement about a real drug target or a real language model.**

The chemistry toolkit in `common/` is a small, transparent fragment-based calculator (exact formula and molecular
weight; approximate logP and TPSA) so everything runs with only NumPy, pandas, SciPy and scikit-learn. For real
compound sets use RDKit for parsing, standardisation and descriptors.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt

python -m pytest                                       # 64 tests
python project-1-bioactivity-data-qc/run_pipeline.py   # writes results/ (report, tables, figures)
python project-2-sar-modeling/run_analysis.py
python project-3-llm-science-eval/run_evaluation.py
```

Each run is deterministic (fixed seeds) and regenerates its `data/` and `results/` folders. Committed example
outputs are included so the reports can be read on GitHub without running anything:

- `project-1-bioactivity-data-qc/results/qc_report.md`
- `project-3-llm-science-eval/results/evaluation_report.md` and `results/feedback/`
- `project-2-sar-modeling/results/` (figures and tables, summarised in the project README)

## Repository layout

```
common/                          shared fragment chemistry + synthetic data generator
project-1-bioactivity-data-qc/   bioqc package, run_pipeline.py, tests, results/
project-2-sar-modeling/          sarmodel package, run_analysis.py, tests, results/
project-3-llm-science-eval/      sciqa_eval package, run_evaluation.py, evaluate_responses.py, tests, results/
.github/workflows/tests.yml      CI: tests + all three pipelines on Python 3.11 and 3.12
```

## Skills demonstrated

- **Data curation and validation:** unit normalisation, assay-range checks, replicate statistics (robust z-scores,
  discordant pairs), duplicate and structure-conflict detection, censored-data handling, audit trail with no silent row loss.
- **Cheminformatics and SAR:** matched molecular pairs, activity cliffs, interaction discovery with multiple-testing
  control, applicability domain, ligand-efficiency-style metrics.
- **Machine learning with honest validation:** random vs group (new-chemotype) cross-validation, bootstrap intervals,
  baseline comparison, permutation importance, prospective ranking on a hidden set.
- **AI-output evaluation:** computed answer keys, error taxonomy, rubric scoring with negation handling, agreement
  statistics (Cohen's kappa), per-responder actionable feedback.
- **Engineering practice:** packaged modules, unit and end-to-end tests, deterministic seeds, CI, documented limitations.

## Limitations (stated up front)

- Synthetic data cannot capture the full messiness of real assays; it shows that the *methods* work and where they fail.
- The property estimates (logP, TPSA) are additive approximations for one chemical series, not validated predictors.
- The project 3 responders are controlled simulators with known error mixes, not language models. The harness can
  grade real model output via `evaluate_responses.py`.
- A keyword rubric under-scores correct answers phrased without the expected terms; the report measures this and
  recommends human review for borderline concept answers.

## License

MIT
