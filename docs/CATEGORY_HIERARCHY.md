# Category Hierarchy and Overlap Controls

## Purpose

GOAT Lab uses nine categories. Several categories intentionally share underlying evidence, so treating all nine as fully independent signals would exaggerate repeated information.

This patch freezes the category hierarchy and the maximum influence of each top-level group before the final model is run.

## Frozen hierarchy

### Performance Arc — 50 percent cap

- Peak
- Prime
- Longevity
- Regular season
- Playoffs

### Basketball Value — 40 percent cap

- Offense
- Defense
- Winning context

### Broader Legacy — 10 percent cap

- Cultural impact

The caps prevent one family of related measurements from dominating only because it contains more subcategories.

## Provisional internal weights

Weights inside each group are currently provisional. They are included so the hierarchy can be tested and audited, but they are not the final preregistered production weights.

The final simulation remains blocked until the preregistration patch freezes them.

## Missing evidence

A missing category is not converted into zero.

Within each group, available evidence is reweighted. Overall coverage decreases in proportion to the missing evidence mass and is exported alongside the diagnostic score.

## Historical overlap audit

The audit calculates pairwise Spearman correlations for the seven categories with a historical career-reference population:

- Peak
- Prime
- Longevity
- Regular season
- Playoffs
- Offense
- Defense

Winning context and cultural impact are excluded because they do not have the same complete historical reference population.

A declared dependency records a known conceptual overlap. An undeclared correlation above the advisory threshold is reported as `high_overlap_advisory`.

High correlation is not by itself a release blocker because correlation does not prove duplicate measurement. Missing reference data or an unavailable correlation is a structural blocker.

## Double-counting controls

The hierarchy addresses repeated information through:

1. Top-level group caps
2. Within-group weights
3. Explicit dependency declarations
4. A historical correlation audit
5. Coverage-aware scoring
6. A final preregistration lock

The hierarchy does not claim that overlapping basketball concepts can be made statistically independent. It makes their shared influence visible and bounded.
