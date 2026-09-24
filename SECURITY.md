# Security policy

## Supported version

This public preview has one supported line: `0.1.x`. It is research software,
not a hosted service or a production security boundary.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting feature if it is enabled for the
repository. Otherwise, contact the repository owner privately. Do not publish
an exploit, a secret, private source material, a continuity database, an
authority checkpoint, or a real session export in a public issue.

Please include the affected version, operating system, reproduction steps,
expected behavior, observed behavior, and a minimal synthetic proof. Remove
tokens, local paths, personal data, model transcripts, and database contents.

## Data boundary

The included retrieval corpus and examples are synthetic and cleared for this
public demonstration. Do not replace them with private correspondence, account
exports, unpublished documents, or personal history in a fork intended for
publication.

The continuity slice can create encrypted SQLite payloads and a separate
authority checkpoint. Its master key must remain outside both files. Runtime
state, keys, checkpoints, databases, model weights, logs, and exports are
ignored by the root `.gitignore`; verify staged files before every commit.

## Network boundary

The demonstration server rejects non-loopback binding and validates host,
origin, content type, request size, and a process-local request token. Those
controls reduce accidental exposure; they are not a substitute for isolation,
authentication, reverse-proxy hardening, or a production threat model. Do not
expose the development server to a LAN or the public internet.
