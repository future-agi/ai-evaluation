# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for a vulnerability.

Email `hello@futureagi.io` with the subject prefix `[security]` and include:

- Affected package, version, commit, or artifact.
- Reproduction steps.
- Expected impact.
- Any known workaround or mitigation.

We will acknowledge reports as soon as possible, triage the issue, and coordinate
fix timing with the reporter when appropriate.

Expected response windows:

- Initial acknowledgement: 3 business days.
- Initial severity assessment: 7 business days.
- Remediation plan or status update: 14 business days for accepted reports.

## Scope

Security reports may cover:

- Secret leakage in artifacts, reports, traces, or logs.
- Unsafe execution behavior in local simulation or optimization flows.
- Dependency or package publishing risks.
- Vulnerabilities in the public Python or TypeScript SDK surfaces.
- Bypass of red-team, policy, approval, or trust-boundary controls.

## Supported Versions

| Version | Supported |
| --- | --- |
| Pre-v1 release branch | Latest maintained release-candidate commit only |
| Older commits | No |

After v1, supported versions should be listed in release notes.
