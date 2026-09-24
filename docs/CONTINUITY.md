# Bounded continuity experiment

The continuity service tests one narrow question: can an explicitly proposed,
visibly consented record survive a process restart and causally change a later
deterministic response relative to a matched no-history condition? A second
slice tests whether a consented correction changes the active interpretation
without deleting or silently rewriting the original record.

## Security model

- SQLite holds encrypted payloads and hash-linked records.
- A separate authority checkpoint detects stale or rolled-back state.
- A base64-encoded 32-byte master key is supplied through an environment
  variable and is not written into the database or checkpoint.
- Proposal terms are canonicalized and hash-bound before consent.
- Consent binds one exact proposal and expires.
- Commits and corrections are atomic and produce retained receipts.
- Independent runtime instances reopen the store for causal checks.

This controlled design does not provide production key custody, hardware-backed
rollback resistance, multi-user authorization, remote service hardening,
backups, or implemented forgetting.

## Run the disposable synthetic evidence scripts

The evaluation scripts create their own temporary state and use only synthetic
sentences:

```powershell
python evaluation/run_continuity_slice.py
python evaluation/run_correction_slice.py
```

Generated evidence goes under the ignored `evaluation/results/` tree. Inspect it
locally, then remove it when no longer needed. Do not commit generated evidence
if you changed the scripts to use real text.

## Guided one-episode CLI

The `seed-continuity` command requires new file paths and refuses to overwrite
existing state. Create the key outside the repository and keep content and
checkpoint locations distinct. The following placeholders are illustrative;
choose secure local paths and do not paste the key into shell history:

```text
SEED_CONTINUITY_MASTER_KEY_B64=<base64 encoded 32-byte key>
seed-continuity --store <private content path> --checkpoint <separate authority path> --owner <local principal> --intention <synthetic intention> --memory <synthetic episode>
```

The CLI prints the canonical proposal terms and requires an exact visible grant
string before committing. Exiting, mistyping, or declining must fail closed.

## Claim boundary

A positive run establishes mechanical record-to-response causality for this
implementation and fixture. It does not show understanding, consciousness,
identity continuity, emotional recognition, or a developed relationship. A
null, adverse, or failed run must remain labeled as such.
