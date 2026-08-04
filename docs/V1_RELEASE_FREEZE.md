# GOAT Lab v1 Release Freeze

## Release deadline

GOAT Lab v1 is frozen for completion within the current release sprint.

No new analytical categories, expert-source families, or methodological
alternatives may be introduced after this freeze unless required to fix a
demonstrable correctness defect.

## Research question

Under a transparent uncertainty-aware framework, how does the available
statistical, playoff, team-success, contextual, defensive, longevity, and
cultural evidence compare Michael Jordan and LeBron James?

## Target players

- Michael Jordan
- LeBron James

## Final category hierarchy

### Performance

- Peak
- Prime
- Longevity
- Regular-season performance
- Playoff performance

### Basketball value

- Offense
- Defense
- Winning context

### Broader legacy

- Cultural impact

## Evidence layers

The final model separates:

1. Quantitative player and team data
2. Awards and recorded recognition
3. Published expert analysis
4. Contextual and cultural evidence

Published expert analysis is labeled `Expert Film Consensus`. It is not
represented as an original blinded film study.

## Double-counting policy

Category construction must prevent the same evidence from being rewarded
multiple times.

Examples requiring explicit controls include:

- Peak versus prime
- Regular season versus longevity
- Playoff performance versus winning context
- Box-score production versus offense
- Awards versus defense
- Championships versus cultural legacy

The final category hierarchy and dependency map must be committed before the
final simulation.

## Scaling policy

The saturated historical-percentile transformation may not be used as the
sole production scale.

The release must compare at least:

- Empirical percentile scaling
- Robust standardized scaling
- Bounded tail-preserving scaling

The production choice must be frozen before observing the final simulation.

## Missing evidence

Missing or incomplete evidence must:

- Never become a zero player grade
- Reduce coverage
- Widen uncertainty
- Remain visible in the dashboard and audit outputs

## Expert-evidence requirements

Expert evidence remains separate from awards.

Multiple articles from one analyst or editorial family count as one
independent source family.

Partial-career evidence may appear in diagnostic views but may not be treated
as full-career evidence.

## Robustness labels

The final result will be classified as:

- `robust`: the same player leads across at least 80 percent of approved
  sensitivity scenarios and the central probability exceeds 60 percent
- `lean`: the same player leads across at least 65 percent of approved
  scenarios or the central probability is between 55 and 60 percent
- `method_dependent`: neither robustness threshold is met
- `indeterminate`: required release gates fail

These labels are methodological descriptions, not statements of objective
basketball truth.

## Final simulation lock

The 250,000-weight final simulation may run only after:

- Scaling is frozen
- Category hierarchy is frozen
- Category weights are frozen
- Uncertainty rules are frozen
- Sensitivity scenarios are frozen
- Tests pass
- Release blockers designated as mandatory are cleared

The final configuration may not be changed in response to which player leads.

## Deferred work

The following may be documented as post-v1 work:

- Additional published expert-source families
- Independent third-party reproduction
- Possession-level proprietary tracking data
- Definitive legal interpretation of source licenses
- Additional historical injury reconstruction where primary records are
  unavailable

## Release outputs

The release must include:

- Frozen configuration
- Final probabilities and intervals
- Category comparison
- Sensitivity matrix
- Data-quality and evidence-coverage report
- Reproduction commands
- Limitations
- Dashboard or static report
- LinkedIn-ready summary
