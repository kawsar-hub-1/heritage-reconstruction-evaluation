# Heritage Reconstruction Evaluation

Research software and reproducible notebooks for **computational and expert-informed evaluation of AI-assisted cultural heritage reconstructions**.

This repository packages code developed around doctoral research on responsible heritage reconstruction, including the **Buddhas of Bamiyan** and a cross-site evaluation at the **Temple of Bel, Palmyra**. It focuses on the evaluation layer: how generated reconstructions can be compared with archival evidence, assessed within semantically meaningful regions, checked for agreement and uncertainty, and compared with expert judgement.

## What the software does

The core Python pipeline can:

- register generated images to an archival reference using ORB features and homography;
- evaluate selected regions of interest (ROIs) rather than only whole-image similarity;
- calculate structural/image measures including SSIM and domain-oriented features;
- parse authorised expert-response tables and calculate inter-rater agreement;
- compare computational scores with expert votes using rank correlation;
- export analysis-ready CSV files and publication-quality figures.

The research notebooks extend this with interactive scoring, perceptual metrics, reliability/sensitivity analysis, multi-criteria ranking and robustness checks.

## Repository structure

```text
heritage-reconstruction-evaluation/
├── src/heritage_eval/          # reusable Python pipeline
├── notebooks/                  # cleaned research notebooks
├── config/rois_example.json    # example ROI configuration
├── examples/                   # synthetic demo-data generator
├── data/                       # data instructions (research data not included)
├── results/                    # generated outputs
├── tests/                      # lightweight regression tests
├── heritage_eval_pipeline.py   # compatibility CLI wrapper
├── requirements.txt
├── pyproject.toml
└── CITATION.cff
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e .
```

Generate public synthetic demo data:

```bash
python examples/generate_demo_data.py
```

Run the pipeline:

```bash
heritage-eval \
  --archival_dir data/demo/archival \
  --baseline_dir data/demo/generated_baseline \
  --rois config/rois_example.json \
  --case Bamiyan \
  --out results/demo
```

On Windows PowerShell, enter the command on one line or use PowerShell's backtick for line continuation.

## Using research data

The original archival imagery, generated reconstruction sets and participant-level response files are **not included** in this public repository. This is intentional: those materials may carry copyright, consent, confidentiality or reuse constraints. The code accepts user-supplied images and, optionally, an authorised Excel response file.

Expected real-data layout is documented in [`data/README.md`](data/README.md).

## Notebooks

See [`notebooks/README.md`](notebooks/README.md) for the role of each notebook. Public notebooks have had execution outputs and machine-specific material removed. Optional PyTorch/PIQ perceptual metrics are available through `requirements-optional.txt`.

## Research context

The software supports research developed through HARF, a human–AI framework for responsible heritage reconstruction. Associated research includes:

- K. Arzomand, T. Kalganova and M. Rustell (2026), **HARF: A Human-AI Collaborative Framework for Responsible Heritage Reconstruction**, *Digital Applications in Archaeology and Cultural Heritage*. DOI: `10.1016/j.daach.2026.e00554`.
- K. Arzomand and T. Kalganova (2026), **Generalising HARF Beyond Bamiyan: A Bounded Cross-Site Evaluation of a Generative-AI Heritage Reconstruction Framework at the Temple of Bel, Palmyra**, *Heritage*.

The wider doctoral evaluation combined computational analysis with heritage-specialist and community assessment. Participant-level datasets are not released here.

## Reproducibility and software quality

- relative paths rather than personal machine paths;
- no API keys or credentials committed;
- synthetic demo data for a public smoke test;
- lightweight automated tests;
- GitHub Actions CI configuration;
- explicit dependency files and software citation metadata.

Run tests with:

```bash
pytest -q
```

## Author

**Kawsar Arzomand, PhD**  
Research interests: artificial intelligence, research software, digital heritage, computer vision, human-AI evaluation and responsible reconstruction.

## Licence

MIT. Research data and third-party heritage imagery are not covered by the software licence and must be used according to their own rights and permissions.
