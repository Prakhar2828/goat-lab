# Category Scaling Audit

## Purpose

The original category table used empirical historical rank percentiles for
seven basketball categories. Both target players sit near the extreme upper
tail of the historical reference population, so several values cluster
between 98 and 100. A small numerical gap on that scale can conceal a much
larger difference in tail position.

This patch audits the problem without selecting a production winner or
unlocking the final simulation.

## Diagnostic scales

The audit compares four declared scenarios.

### `historical_percentile`

The existing empirical rank percentile. It is intuitive and distribution
free, but it compresses elite players when both are near the top of the same
reference population.

### `normal_score_tail`

The empirical percentile is converted through an inverse-normal transform
and mapped back to a bounded 0-100 display scale. This expands differences in
the tails.

### `bounded_logit_tail`

The empirical percentile is transformed through bounded log odds. The
function is monotonic, finite at 0 and 100 after clipping, and preserves more
elite-tail separation than the raw percentile.

### `robust_mad_reference`

The raw category value is standardized against the historical median and
median absolute deviation. The robust z-score is passed through a bounded
hyperbolic-tangent mapping.

When MAD is zero, the implementation falls back in order to:

1. interquartile-range scale,
2. population standard deviation,
3. a constant-reference safeguard.

The fallback method is recorded in every output row.

## Native evidence scales

`winning_context` and `cultural_impact` currently use their own evidence
scales because there is no complete historical player reference population
for either category. The scaling audit preserves them unchanged in all four
scenarios and labels them `native_evidence_scale`.

## Outputs

Running:

```bash
.venv/bin/python scripts/run_category_scaling_audit.py
```

writes:

- `data/processed/category_scaling_comparison.parquet`
- `data/processed/category_scaling_saturation_audit.parquet`
- `data/processed/category_scaling_audit.json`
- `data/processed/category_scaling_audit.txt`

## Release rule

This audit does not select the production scale. The production scale must be
frozen during preregistration before the final 250,000-weight simulation.

The final simulation remains blocked after this patch.
