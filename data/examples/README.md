# ADD Example Corpus

This folder contains a tiny illustrative corpus for exercising the ADD evidence schema and local loader.

The cases are fictional, synthetic examples. They are not real incidents, not representative samples, and not calibration data. The source is marked `experimental` and `tier_4`, so the corpus is excluded from `EvidenceCorpus.calibration_evidence()` by default.

Files:

- `sources.json`: source registry metadata for the illustrative corpus.
- `evidence.jsonl`: one synthetic evidence unit per line.

Example use:

```bash
.venv/bin/python - <<'PY'
from core_model.evidence_io import load_corpus

corpus = load_corpus("data/examples/sources.json", "data/examples/evidence.jsonl")
print(corpus.coverage_summary())
print(corpus.calibration_evidence())
PY
```

The expected calibration evidence output is empty unless a caller explicitly opts into experimental sources elsewhere. This is intentional.
