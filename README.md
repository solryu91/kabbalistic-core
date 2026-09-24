# Kabbalistic Core — public research edition

Kabbalistic Core is a deterministic, inspectable cognitive graph organized by
a Kabbalistic topology. It turns an intention into typed perspectives, directed
peer observations, an explicit integration step, a bounded symbolic return,
and a traceable response. A local interface can optionally use a compatible
Qwen3 8B model for bounded proposals while retaining a visibly labeled
deterministic fallback.

This repository is a clean public-edition export. It contains no private Git
history, private corpus, personal archive, cloud-document locator, model weight,
continuity database, authority checkpoint, or real session trace. The bundled
corpus and evaluation inputs are project-authored synthetic fixtures.

## What it does

- Routes one intention through an explicit Tree-of-Life graph.
- Maintains three distinct proposal roles: Form, Flow, and Accord.
- Requires all three first-pass views before six directed peer observations.
- Integrates at Tiferet without allowing a majority to override hard bounds.
- Records permissions, transitions, evidence bindings, state hashes, and stop
  reasons instead of presenting hidden reasoning.
- Offers three project-owned functional kernels: `clear_sight`, `liberation`,
  and `regeneration`.
- Retrieves exact excerpts from a versioned, hash-bound synthetic corpus.
- Compares the graph with a budget-matched neutral three-perspective pipeline.
- Includes a bounded encrypted continuity experiment with explicit proposal,
  consent, commit, correction, and causal comparison records.

## What it does not claim

This is not a consciousness detector, a copied person, a general-purpose agent,
an autonomous authority, or proof that software exhausts Kabbalah or mystical
experience. Symbolic intensity is not evidence. Retrieved text cannot grant
tools or permissions. A fluent response is not proof of correctness, identity,
growth, or personhood.

The continuity code demonstrates narrow record-to-response causality with
synthetic inputs. It is not a general relationship-memory product. Forgetting,
production key management, multi-user isolation, remote deployment, and a
complete threat model are outside this preview.

## Repository map

```text
src/kabbalistic_core/   graph, engine, trace, retrieval, evaluation, continuity
tests/                  Python contract and causal-slice tests
ui/                     local inspection interface and rendered HTML tests
evaluation/             synthetic comparison definitions and local runners
examples/corpus_sources project-authored source documents for the public corpus
docs/                   architecture, boundaries, evaluation, and operation notes
```

## Quick start: deterministic CLI

Requirements: Python 3.12 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
kabbalistic-core --intention "Choose one reversible next step" --symbol "threshold" --kernels clear_sight,regeneration --no-memory-proposal
```

The CLI writes paired trace artifacts to `runs/` by default. That directory is
ignored because a real intention or feedback record may be private. Use
`--output-dir` to select another local-only location.

On macOS or Linux, activate the virtual environment with
`source .venv/bin/activate` and use the same Python and CLI commands.

## Local inspection interface

Requirements: Python 3.12+, Node.js 22.13+, and pnpm 10.

```powershell
python -m pip install -e .
Set-Location ui
pnpm install --frozen-lockfile
Set-Location ..
.\start-seed.ps1 -CheckOnly
.\start-seed.ps1
```

The launcher binds the backend and interface to loopback only. If a compatible
local Qwen3 8B model service and model file are available, the launcher can use
them. Otherwise the application exposes the deterministic fallback as such.
It does not download dependencies or model weights at runtime.

For a portable two-terminal setup on another operating system:

```text
Terminal 1: python -m kabbalistic_core.poc_server
Terminal 2: pnpm --dir ui run dev
```

Then open `http://127.0.0.1:3000` locally.

## Public synthetic corpus

The default corpus is classified `public-demo-safe` with publication consent
recorded. It contains twelve embedded excerpts drawn from the three Markdown
files in `examples/corpus_sources/`. Document and excerpt SHA-256 values make
the exact input field inspectable. All three documents were written for this
public edition; they are not excerpts from a private archive.

To substitute a corpus, preserve the schema and declared hashes. Public runtime
mode refuses a corpus unless its classification is `public-demo-safe` and its
publication consent is `recorded`. A label is not a rights review: only add
material you own or are independently authorized to redistribute.

## Evaluation harness

The evaluation module supports hash-bound paired outputs, deterministic balanced
blinding, complete score sheets, sealed scores, and a preregistered decision
rule that preserves null results.

The included JSON definitions are project-authored synthetic public fixtures.
They exercise the public evaluation API and protocol evolution without
publishing or summarizing any nonpublic research run. Generate new local results
from the public fixtures instead of treating them as historical evidence.

## Continuity experiment

The continuity runner creates disposable local stores from synthetic inputs and
tests one consented episode followed by a bounded correction. The data key is
external to both the SQLite content store and the authority checkpoint.

Read `docs/CONTINUITY.md` before running it. Never commit a database, checkpoint,
master key, real relationship history, or generated evidence containing user
input. The ignore rules are a backstop, not permission to handle sensitive data
carelessly.

## Verification

```powershell
python -m unittest discover -s tests -v
pnpm --dir ui run lint
pnpm --dir ui run test
```

The interface test builds the application and exercises the server-rendered
HTML. GitHub Actions runs the same Python and interface checks on pushes and
pull requests.

## Security and publication boundary

Read `SECURITY.md` and `docs/PUBLICATION_BOUNDARY.md` before publishing a fork.
The development server is local-only and not suitable for internet exposure.
The repository deliberately ignores private content, databases, checkpoints,
environment files, keys, logs, exports, build products, and model weights.

## License

This preview is **not open source yet**. The temporary `LICENSE` reserves all
rights until the repository owner makes a deliberate software and content
license choice. Third-party dependencies remain under their own licenses.
