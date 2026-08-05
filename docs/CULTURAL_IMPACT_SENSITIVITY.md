# Cultural-Impact Sensitivity Freeze

## Purpose

The version 1 cultural-impact score combines two different kinds of evidence:

- common-window Wikimedia attention;
- a sourced historical and institutional rubric.

The production blend is 20% attention and 80% rubric. That choice is
normative, so this audit freezes a reasonable sensitivity grid before the
final simulation.

## Evidence coverage

Both Michael Jordan and LeBron James have complete Wikimedia coverage over
the same comparison window. Both also have all four required rubric
dimensions.

Complete input coverage does not make the weighting choice objective.
Wikimedia measures modern digital attention beginning in 2015, while the
rubric attempts to represent longer-term commercial, basketball-cultural,
media, and institutional influence.

## Frozen grid

The audit evaluates:

- five attention-versus-rubric weights;
- four attention-component weighting schemes;
- four rubric-dimension weighting schemes.

This produces 80 scenarios and 160 player rows.

The configured 20/80 blend is included as an explicit baseline and must
reproduce the existing cultural scores exactly.

## Interpretation

The audit reports:

- the winner under every scenario;
- the LeBron-minus-Jordan gap;
- the closest scenarios;
- the attention weight at which the configured component blends tie.

A winner change within the grid means the cultural-impact ordering is not
robust to reasonable weighting choices. This is a methodological finding,
not a reason to select a preferred scenario after seeing the result.

## Model treatment

All sensitivity scenarios are diagnostic only:

- `PRIMARY_MODEL_ELIGIBLE` is false;
- `ADDITIONAL_CENTRAL_WEIGHT` is zero;
- the production cultural scores are not overwritten;
- the final simulation remains blocked until the release freeze is complete.

The final release must describe cultural impact as a sourced but
value-sensitive component rather than a purely objective measurement.
