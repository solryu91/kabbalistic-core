# Public-edition boundary

## Included

- The deterministic Python graph, models, permissions, trace validation,
  retrieval, evaluation, local proof-of-concept service, server, and CLIs.
- The bounded continuity and correction implementation.
- The complete Python test suite, including security and causal-slice tests.
- The interface source, configuration, lockfile, accessibility checks, and
  rendered HTML tests.
- Three project-authored synthetic corpus sources and their twelve embedded,
  content-hash-bound excerpts.
- Synthetic evaluation definitions and local evidence runners.
- Root security, contribution, citation, license, ignore, changelog, and CI
  files, plus public architecture and operating notes.

## Deliberately excluded

- The source repository's `.git` directory, branches, commits, tags, and remote
  configuration. This edition should begin with fresh public history.
- Every nonpublic source document, excerpt, locator, content digest tied to a
  nonpublic source, authorship ledger, internal review, private planning note,
  and publication-consent discussion.
- Person- or franchise-specific inspiration provenance from shipped data,
  interface copy, code contracts, and tests.
- Results or summaries derived from nonpublic research runs, including paired
  model outputs, blind packets, condition keys, score sheets, and evaluator
  notes.
- Real session traces, intentions, feedback, exported bundles, logs, databases,
  authority checkpoints, keys, backups, and model weights.
- Dependency directories, build output, caches, and generated evidence.

## Before creating a public repository

1. Confirm the public attribution in `LICENSE` and `CITATION.cff`.
2. Choose a long-term code license and a separate content license if public
   reuse should be permitted. Until then, the all-rights-reserved notice stays.
3. Run the Python and interface checks from `README.md`.
4. Search the complete directory for personal names, email addresses, account
   identifiers, cloud locators, secrets, and absolute local paths.
5. Review every staged file. Confirm that no ignored runtime artifact was added
   with a force option.
6. Publish only from this clean public edition. Do not connect or push the
   original private repository.
7. Enable private vulnerability reporting and branch protection where available.

This manifest documents the intended boundary; it is not a substitute for the
owner's final legal, privacy, and security review.
