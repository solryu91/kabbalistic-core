# Contributing

Thank you for the interest. This is a curated public preview under an
all-rights-reserved notice while the repository owner chooses a long-term
license. Unsolicited code or content contributions should not be submitted yet.

Discussion, reproducible bug reports, and security reports are welcome. Use a
minimal synthetic example and remove personal data, private archives, API keys,
model transcripts, local file paths, databases, checkpoints, and generated run
artifacts.

If the owner later opens contributions, a proposed change should:

- preserve deterministic replay and visible execution labels;
- keep symbolic text separate from runtime authority;
- add or update tests for changed behavior;
- use only project-authored, synthetic, or independently rights-cleared data;
- preserve unfavorable, failed, and null evaluation outcomes;
- pass the Python suite and interface lint/build/render checks; and
- avoid adding dependencies without a clear need and license review.

Python code targets Python 3.12 or newer. The interface targets Node.js 22.13 or
newer and uses the committed pnpm lockfile.
