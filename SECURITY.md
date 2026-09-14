# Security Policy

## Supported versions

Xazz is distributed both as source and as pre-built binaries. Security fixes are
applied to the latest release on the `main` branch.

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✅        |
| < 0.3   | ❌        |

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Instead, report it
privately through GitHub's [private vulnerability reporting](https://github.com/x1zzdev/Xazz/security/advisories/new)
(the **Report a vulnerability** button under the repository's **Security** tab).

Please include:

- A description of the issue and its impact
- The affected component (CLI, compiler, exec/runner, server, Visual IDE)
- A minimal `.xzz` source or request that reproduces the issue
- The Xazz version (`xazz --version`) and your operating system
- Any suggested fix or mitigation, if you have one

We aim to acknowledge a report within **5 business days**, provide an initial
assessment within **10 business days**, and coordinate a disclosure timeline with
you. Please give us a reasonable window to release a fix before public
disclosure.

## Scope

Xazz is a local compiler and pipeline runtime. In scope:

- The compiler pipeline (`xazz-core`, `xazz-compiler`): parser, type checker,
  Typed IR, policy engine.
- The execution layer (`xazz-exec`, `xazz-runner`): Polars/Burn lowering,
  process isolation and timeout handling.
- The server (`xazz-server`): REST API, auth/multi-tenant handling, audit chain,
  the optional on-prem sLM hook.
- The static policy guardrail: bypasses of PII/secret/prompt rules, or false
  negatives that allow a blocked literal through.
- The audit chain: any way to tamper with the SHA-256 chain without detection.
- The bundling/packaging: release archives, Docker image, VS Code extension.

## Out of scope / known boundaries

These are documented design boundaries, not vulnerabilities:

- **Process isolation is not an OS sandbox.** `xazz run` spawns `xazz-runner` as
  a subprocess with an execution timeout; it does not restrict filesystem or
  network access of the executed pipeline.
- **The policy guardrail is a static, fail-closed text/rule scanner**, not a
  formal verifier. It reduces risk; it does not prove absence of sensitive data.
- **Differential privacy** provides ε/δ composition accounting for the queries
  Xazz runs; it is not a claim about the upstream data collection.
- **The licensing marker** is a legal/contract marker, not a technical lock —
  see [docs/design/licensing.md](docs/design/licensing.md).

## Handling

Security reports are triaged by the maintainers, fixed on a private branch when
appropriate, and released as a patch version with an advisory in the
[GitHub Security tab](https://github.com/x1zzdev/Xazz/security). Dependency
vulnerabilities are tracked by `cargo deny check advisories` (see
[`deny.toml`](deny.toml)) in CI.