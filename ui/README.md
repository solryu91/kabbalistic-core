# Local inspection interface

This interface renders the public synthetic demonstration. It talks only to the
loopback backend, displays the outward response and exact retrieved excerpts,
and exposes typed process records, boundaries, model/fallback labels, and the
neutral comparison condition.

It is not a hosted application, an account system, or a durable-memory surface.
Do not deploy the development server to a LAN or the public internet.

```powershell
pnpm install --frozen-lockfile
pnpm run dev
pnpm run lint
pnpm run test
```

The complete Windows launcher lives at the repository root. Run
`start-seed.ps1 -CheckOnly` to inspect local prerequisites without starting a
process.
