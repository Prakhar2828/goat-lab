# Defense Evidence and Recognition

## Purpose

GOAT Lab separates three defensive evidence layers:

1. Historical box-score and season-value evidence
2. Recorded defensive awards
3. Published expert film analysis

The layers are not interchangeable. Awards measure recorded recognition, while
expert analysis attempts to describe defensive skill and role. Neither is
treated as a replacement for the historical statistical layer.

## Awards source

The pipeline reads:

```text
data/interim/player_awards.parquet
```

This file is produced by the existing NBA player-awards ingestion step.

Recognized defensive awards are normalized into:

- Defensive Player of the Year
- All-Defensive First Team
- All-Defensive Second Team
- All-Defensive Team when the team level is unavailable

Duplicate player-season-award rows are removed before scoring.

## Awards points

The fixed points are:

| Recognition | Points |
|---|---:|
| Defensive Player of the Year | 5.0 |
| All-Defensive First Team | 2.0 |
| All-Defensive Second Team | 1.0 |
| All-Defensive Team, unspecified | 1.5 |

The display score is:

```text
100 × defensive award points / 25
```

and is clipped to 0–100.

The 25-point benchmark is fixed before the final simulation. It is not
calibrated to whichever target player has the larger award total.

Awards receive only the existing awards share of the defense category. They
are not inserted into peak, prime, playoffs, winning context, or cultural
impact.

## Expert film eligibility

The expert film framework currently contains phase-specific evidence. A film
consensus row can affect the model only when:

```text
PRIMARY_MODEL_ELIGIBLE == true
```

Phase-detail-only rows remain visible in diagnostics but do not become a
player score. Missing or ineligible film evidence remains missing; it is never
converted into zero.

At the current release checkpoint, expert film contributes diagnostic
coverage and limitations but does not enter the defense category unless the
eligibility gate is satisfied.

## Category integration

The existing defense construction uses these nominal component weights:

- Historical defensive box evidence: 50%
- Eligible expert film evidence: 35%
- Defensive awards: 15%

Only available, eligible components are reweighted. This prevents missing film
evidence from being treated as poor defense.

Manual category inputs remain override fields for emergency corrections, but
the ordinary awards score now comes from the reproducible processed evidence
file:

```text
data/processed/defense_evidence_scores.parquet
```

## Audit outputs

The defense audit writes:

```text
data/processed/defense_evidence_scores.parquet
data/processed/defensive_awards_normalized.parquet
data/processed/defense_evidence_audit.json
data/processed/defense_evidence_audit.txt
```

The audit records:

- Recognized award rows
- Award counts and points
- Award score
- Film coverage
- Primary-eligible film rows
- Whether film entered the model
- Release blockers

## Limitations

Awards depend on historical voting, media attention, positional conventions,
team success, and the defensive statistics available in each era.

All-Defensive selections and Defensive Player of the Year voting should not be
read as direct measurements of possession-level defensive value.

The expert film register is not an original blinded film study. It is a
systematic synthesis of published analysis, deduplicated by source family.

The final simulation remains blocked until the production scale, weights,
uncertainty rules, and sensitivity scenarios are frozen.
