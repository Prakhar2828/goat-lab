# Expert Evidence Methodology

## Purpose

GOAT Lab supplements statistical evidence with a systematic review of published expert film analysis.

This component is called **Expert Film Consensus**. It is not an original blinded film study and must never be described as one.

## Unit of evidence

The fundamental unit is a sourced analytical claim.

Every claim must identify:

- The source
- Analyst and publication
- Player
- Career phase
- Offensive or defensive dimension
- Direction and strength
- Supporting location
- Evidence type
- Confidence
- Limitations
- Review status

No unsupported narrative is converted into a player score.

## Source families

Multiple publications from the same analyst, outlet, series, or research project may share methods and evidence.

GOAT Lab therefore aggregates claims within `SOURCE_FAMILY` before calculating consensus. Ten articles from one family do not count as ten independent expert opinions.

## Source-quality score

Sources are evaluated before their conclusions are considered.

The maximum score is 19:

| Criterion | Maximum |
|---|---:|
| Analyst expertise | 3 |
| Film specificity | 3 |
| Methodology transparency | 3 |
| Sample disclosure | 2 |
| Statistical support | 2 |
| Career coverage | 2 |
| Balanced positive and negative evidence | 2 |
| Source independence | 2 |

Tiers:

- Tier A: 14–19
- Tier B: 10–13
- Tier C: 6–9
- Excluded: below 6

A favorable conclusion does not increase source quality.

## Claim scale

Claim directions:

- `major_strength`
- `strength`
- `mixed`
- `limitation`
- `major_limitation`

Each claim also receives:

- Strength from 1 to 3
- Confidence from 0 to 1
- A review status

Only `verified` and `verified_with_qualification` claims from non-excluded sources enter consensus.

## Consensus

Claims are first combined within source family. Family-level values are then combined across independent families using source-quality weights.

The output includes:

- Consensus score
- Lower and upper source-family quantiles
- Source-family disagreement
- Number of source families
- Number of Tier A families
- Number of accepted claims
- Evidence eligibility

## Primary-model eligibility

A career-level dimension may enter the primary model only when:

1. At least three independent source families cover it.
2. At least one source family is Tier A.
3. Both Michael Jordan and LeBron James have coverage.
4. The dimension is configured as primary-eligible.
5. All included claims have accepted review statuses.

Career-phase evidence may be displayed even when it is not eligible for the primary career score.

## Missing evidence

Missing evidence never becomes a zero.

Insufficient coverage creates a release blocker or widens uncertainty. It does not automatically reduce either player's score.

## Awards

Awards and contemporary recognition remain separate from Expert Film Consensus. They must not be copied into expert-film claims merely to increase apparent agreement.

## Reproducibility

The public release must include:

- `data/manual/expert_sources.csv`
- `data/manual/expert_claims.csv`
- `data/manual/expert_analysis_dimensions.csv`
- Generated source-quality output
- Generated consensus output
- Generated blocker output
- This methodology document

The project should publish analytical summaries rather than copyrighted video, screenshots, or long source excerpts.
