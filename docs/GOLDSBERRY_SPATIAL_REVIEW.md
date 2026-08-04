# Goldsberry Spatial Scoring Review

## Scope

This review adds structured claims from:

- `GOLD_MJ_SCORING_2020`
- `GOLD_LBJ_ATLAS_2018`
- `GOLD_LBJ_SCORING_RECORD_2023`

All three documents belong to one source family:

```text
goldsberry_espn_spatial
```

Multiple articles from the same analyst family do not count as
multiple independent expert families.

## Evidence type

The claims are classified as:

```text
spatial_scoring_analysis
```

They use shot-location data, shot charts, scoring splits, and tactical
interpretation. They are not labeled as original GOAT Lab film grades,
and `FILM_EXAMPLES_PRESENT` remains false.

## Temporal coverage

Jordan's quantified shot-location evidence is concentrated in the
1996-97 and 1997-98 seasons. Career-evolution claims are labeled only
through 1998.

LeBron's two reviews end in 2018 and 2023 respectively. Neither is
treated as complete coverage of his eventual career.

No Goldsberry claim uses the primary `career` phase.

## Dimensions included

Claims are limited to dimensions directly supported by the articles:

- Rim pressure and finishing
- Midrange creation
- Half-court creation
- Post scoring
- Shooting gravity, with qualification
- Passing execution
- Advantage creation
- Off-ball value
- Scalability
- Playoff resilience, with qualification
- Role adaptability

No defensive claim is inferred from these offensive scoring analyses.

## Eligibility

These rows can expand phase-level evidence and measure agreement with
the Thinking Basketball anchor reviews.

They cannot enter the primary expert-film score because:

- They are partial-career or phase-specific.
- They represent only one additional independent source family.
- Several dimensions still lack matched evidence for both players.
- The articles are spatial scoring analyses rather than complete
  two-way film studies.

## Reproducibility

Run:

```bash
.venv/bin/python scripts/import_goldsberry_claims.py
.venv/bin/python scripts/build_expert_claim_register.py
.venv/bin/python scripts/audit_expert_evidence.py
```

The importer is idempotent: it replaces only claims from the three
registered Goldsberry source IDs and preserves claims from every other
source family.
