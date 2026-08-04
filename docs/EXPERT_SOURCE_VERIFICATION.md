# Expert Source Verification

GOAT Lab verifies registered expert-analysis sources before admitting their claims into the evidence model.

## Automated checks

The verifier records:

- Fetch method
- HTTP status
- Redirect destination
- Observed title
- Observed analyst
- Observed publication
- Observed publication date
- Expected-title match
- Expected-analyst match
- Expected-date match
- Player-coverage check
- Analytical-content check
- SHA-256 content fingerprint
- Response size
- Fetch errors

YouTube sources use structured oEmbed metadata because direct video-page HTML may not be consistently available to automated clients.

## Manual review

Automated metadata checks cannot determine whether a source's reasoning is persuasive or whether every registered use is justified.

Every source therefore retains a separate review status:

- `pending`
- `verified`
- `verified_with_qualification`
- `rejected`

Only `verified` and `verified_with_qualification` sources may support verified claims.

## Content changes

Each verification snapshot includes a SHA-256 fingerprint. A changed fingerprint does not automatically invalidate a source, but it requires review when the source is used in a release.

## Migrated pages

A URL path is not treated as publication metadata. Some sites migrate older content into newer URL structures. The displayed or structured publication date is compared with the registered date.

## Release behavior

The release audit blocks publication when:

- A registered source has no verification row.
- Automated verification fails.
- Human review remains pending.
- A verified claim references an unverified source.

## Qualified manual overrides

An automated failure does not always mean that a source is invalid. Examples include:

- A publisher returning HTTP 403 to automated clients
- A video page exposing metadata but not its full analytical transcript
- A page omitting author or publication-date metadata
- A migrated page whose visible metadata differs from its URL structure

An automated failure may be accepted only as `verified_with_qualification`.

The qualified review must record:

- Reviewer
- Review timestamp
- Specific reason for the automated failure
- What source identity and analytical content were independently confirmed
- Any remaining limitation

A plain `verified` status cannot override an automated failure. This prevents undocumented manual approval from bypassing the audit.
