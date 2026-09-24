# Public evaluation fixtures

This directory contains public synthetic definitions and local runners for the
paired graph-versus-neutral evaluation and the bounded continuity experiments.

The three `preregistered_v*.json` files are intentionally labeled
`synthetic_public_fixture`. They preserve schema, blinding, sealing, and
decision-rule tests without presenting sanitized inputs as the historical
preregistration. Each adjacent `.sha256` file contains the stable canonical
object hash used by the test suite.

`capture_preregistered.py` can capture a newly frozen evaluation against a local
compatible model. Use a new output directory, keep the condition key separate
from the evaluator, and never replace a failed prompt after viewing outcomes.
`pilot_local.py` performs a local comparison-eligibility smoke check.

`run_continuity_slice.py` and `run_correction_slice.py` create disposable
synthetic state, emit evidence under `evaluation/results/`, and preserve null or
failed outcomes. Generated results are ignored by default.

This public edition contains no results or summaries derived from nonpublic
research runs. Generate new local evidence from the synthetic fixtures.
