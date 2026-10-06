# Data

The original doctoral research data are **not included** in this public repository. This avoids redistributing participant-level responses, restricted archival material, or generated images whose reuse conditions may differ.

Expected real-data layout:

```text
data/
├── archival/                 # archival/reference image(s)
├── generated_baseline/       # reconstruction candidates
└── expert_responses.xlsx     # optional; use only if you are authorised to share/use it
```

For a fully public smoke test, run `python examples/generate_demo_data.py` to create synthetic images under `data/demo/`.
