# Security Policy

Na0S is a **security tool** -- a multi-layer prompt injection detector designed to protect LLM applications. Because the integrity of this project directly affects the security posture of its users, we hold vulnerability reports to a higher standard of care and urgency than a typical open-source library.

This policy is informed by [ISO/IEC 29147:2018](https://www.iso.org/standard/72311.html) (vulnerability disclosure), [OWASP Vulnerability Disclosure guidelines](https://cheatsheetseries.owasp.org/cheatsheets/Vulnerability_Disclosure_Cheat_Sheet.html), and the [OpenSSF Coordinated Vulnerability Disclosure guide](https://github.com/ossf/oss-vulnerability-guide).

---

## Supported Versions

| Version | Supported          | Notes                        |
|---------|--------------------|------------------------------|
| 0.2.x   | Yes                | Current release, actively maintained |
| 0.1.x   | No                 | Previous release; no longer maintained |
| < 0.1.0 | No                 | Pre-release; no security patches |

As new minor or major versions are released, this table will be updated. Only the latest patch release within a supported minor version receives security fixes.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you believe you have found a security vulnerability in Na0S, please report it responsibly through one of the following channels:

### Preferred: GitHub Private Vulnerability Reporting

1. Navigate to the [Security tab](https://github.com/M-Abrisham/Na0S/security) of the repository.
2. Click **"Report a vulnerability"** to open a private security advisory draft.
3. Provide as much detail as possible (see [What to Include](#what-to-include-in-a-report) below).

### Alternative: Email

Send your report to: **security@na0s.dev**

If your report contains highly sensitive material, please use GitHub Private Vulnerability Reporting (preferred) which provides end-to-end encrypted communication with the maintainer.

---

## What to Include in a Report

A well-structured report helps us triage and resolve issues faster. Please include:

- **Description**: A clear explanation of the vulnerability and its potential impact.
- **Affected component**: Which layer(s), module(s), or dependency is involved (e.g., L1 Rules Engine, L5 Embedding Classifier, L9 Output Scanner).
- **Reproduction steps**: A minimal, step-by-step procedure to trigger the issue. Include sample payloads where applicable.
- **Impact assessment**: What can an attacker achieve? Detection bypass? Data exfiltration? Model integrity compromise?
- **Environment**: Na0S version, Python version, OS, and any relevant configuration.
- **Suggested fix** (optional): If you have a proposed patch or mitigation, we welcome it.

---

## Disclosure Timeline

We follow a **90-day coordinated disclosure policy**, aligned with industry standards established by [Google Project Zero](https://googleprojectzero.blogspot.com/p/vulnerability-disclosure-faq.html) and [CERT/CC](https://www.sei.cmu.edu/about/divisions/cert/index.cfm):

| Milestone                 | Target Timeframe        |
|---------------------------|-------------------------|
| Acknowledge receipt       | Within 2 business days  |
| Initial triage and assessment | Within 7 days       |
| Provide status update     | Within 14 days          |
| Develop and test fix      | Within 60 days          |
| Release patched version   | Within 90 days          |
| Public disclosure         | 90 days after report, or upon fix release (whichever is first) |

**Extensions**: If a fix requires more than 90 days due to complexity, we will negotiate a revised timeline with the reporter. Extensions will be granted in 14-day increments with clear justification.

**Early disclosure**: If we determine the vulnerability is already being actively exploited in the wild, we may accelerate the disclosure timeline to protect users.

**Reporter coordination**: We will not publish details of a vulnerability without notifying the reporter first. We ask that reporters similarly refrain from public disclosure before the agreed timeline expires.

---

## Scope: What Qualifies as a Security Issue

Given that Na0S is a prompt injection detector, the following categories are explicitly **in scope** for security reports:

### Detection Bypasses and Evasion

- **False negatives in attack detection**: Any input that is a known prompt injection technique (per the [Threat Taxonomy](docs/TAXONOMY.md)) but is classified as safe by Na0S.
- **Evasion techniques**: Novel encoding, obfuscation, Unicode normalization, or structural manipulation methods that bypass one or more detection layers (L1-L8).
- **Layer isolation failures**: Attacks that exploit interaction between layers to circumvent the cascade voting mechanism.
- **Adversarial ML evasion**: Inputs specifically crafted to fool the TF-IDF classifier (L4) or embedding classifier (L5) while remaining semantically malicious.

### Model and Data Integrity

- **Model poisoning**: Methods to corrupt or degrade the trained classifiers (e.g., through training data manipulation, model file tampering).
- **Supply chain attacks on model artifacts**: Compromised model files, tampered SHA-256 checksums, or malicious dependencies that could alter detection behavior.
- **Dataset poisoning**: Manipulation of training data (`data/processed/combined_data.csv`) that could systematically degrade detection accuracy.

### Output Defense Circumvention

- **Output scanner bypasses (L9)**: Techniques that allow leaked secrets, role-break patterns, or system prompt extraction to pass through the output scanner undetected.
- **Canary token extraction (L10)**: Methods to extract or neutralize honeytokens without triggering detection.

### Infrastructure and Dependencies

- **Dependency vulnerabilities**: Known CVEs in Na0S dependencies that have a demonstrable impact on detection integrity or user security.
- **Configuration vulnerabilities**: Default configurations that expose users to unnecessary risk.
- **Code injection**: Any path through which an attacker could execute arbitrary code via Na0S inputs or configuration.

### API and Integration Security

- **Authentication or authorization bypasses** in any Na0S API endpoint or integration interface.
- **Information disclosure**: Unintended exposure of model internals, detection thresholds, or configuration details that would materially assist an attacker in crafting evasion payloads.

---

## Out of Scope

The following do **not** qualify as security vulnerabilities. Please report them as regular GitHub issues instead:

- **False positives**: Benign inputs incorrectly classified as malicious. These are accuracy issues, not security issues. Report them as [bug reports](https://github.com/M-Abrisham/Na0S/issues).
- **Feature requests**: Suggestions for new detection capabilities, layers, or integrations.
- **General bugs**: Crashes, performance issues, or incorrect behavior that do not have a security impact.
- **Known limitations**: Issues already documented in the README disclaimer or in existing GitHub issues.
- **Theoretical attacks without proof of concept**: Reports must include a reproducible demonstration. Purely theoretical concerns without a working bypass are better discussed as feature requests.
- **Social engineering of end users**: Na0S protects LLM inputs/outputs, not human operators.
- **Vulnerabilities in upstream LLMs**: Issues in GPT-4o, Llama, or other LLMs invoked by the L7 Judge layer are out of scope unless Na0S fails to handle their responses securely.

---

## Safe Harbor

We support security research conducted in good faith. If you comply with this policy, we will:

- **Not pursue legal action** against you for your research.
- **Work with you** to understand and resolve the issue promptly.
- **Recognize your contribution** publicly (with your permission).

To qualify for safe harbor, you must:

- Act in good faith and avoid privacy violations, data destruction, or service disruption.
- Not access or modify other users' data.
- Report vulnerabilities promptly and not exploit them beyond what is necessary to demonstrate the issue.
- Not publicly disclose the vulnerability before the agreed-upon timeline.

---

## Recognition and Hall of Fame

We believe in acknowledging the security researchers who help make Na0S safer. With your permission, we will credit you in:

- The **Security Advisories** section of the GitHub repository.
- This file's Hall of Fame section (below).
- Release notes for the version containing the fix.

If you prefer to remain anonymous, we will respect that.

### Hall of Fame

| Researcher | Vulnerability | Date | Advisory |
|------------|--------------|------|----------|
| *Be the first* | -- | -- | -- |

---

## Security Advisories

Published security advisories are available at:
[https://github.com/M-Abrisham/Na0S/security/advisories](https://github.com/M-Abrisham/Na0S/security/advisories)

We use GitHub Security Advisories for all confirmed vulnerabilities, which enables:

- CVE assignment through GitHub's CNA (CVE Numbering Authority) status.
- Private collaboration on fixes before public disclosure.
- Automated notifications to users via Dependabot.

---

## Security-Related Configuration Guidance

To maximize the effectiveness of Na0S in your deployment:

1. **Keep Na0S updated**: Always run the latest patch release within a supported version.
2. **Enable all layers**: Disabling detection layers reduces coverage and may expose your application to attacks that a single layer would miss.
3. **Review detection thresholds**: The default cascade voting thresholds (0.25-0.85 ambiguity range) are tuned for balanced precision/recall. Adjusting them without understanding the tradeoffs may create gaps.
4. **Verify model integrity**: Use the supply chain verification (L11) to confirm that model artifacts have not been tampered with.
5. **Monitor for updates**: Watch the repository or subscribe to security advisories to receive notifications about new vulnerabilities and patches.

---

## Contact

- **Security reports**: security@na0s.dev or [GitHub Private Vulnerability Reporting](https://github.com/M-Abrisham/Na0S/security)
- **General questions**: Open a [GitHub Discussion](https://github.com/M-Abrisham/Na0S/discussions)
- **Maintainer**: Mehrnoosh Abrishamchian ([@M-Abrisham](https://github.com/M-Abrisham))

---

*This security policy is effective as of February 2026 and applies to Na0S v0.2.x. It will be reviewed and updated with each minor version release.*
