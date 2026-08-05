# GOAT Lab v1 Release QA

The public release must be validated without rerunning `goatlab train-models`.

## Local release QA

```bash
make release-qa
```

Equivalent commands:

```bash
ruff check src tests app scripts
pytest -q
python scripts/verify_public_release.py
python scripts/build_release_assets.py --verify
find app -type f -name '*.py' -print0 | xargs -0 python -m py_compile
git diff --check
```

## Docker smoke test

```bash
docker build --no-cache -t goat-lab:v1 .
docker run --rm -d --name goat-lab-v1 -p 8501:8501 goat-lab:v1
sleep 8
curl --fail --silent --show-error http://localhost:8501/_stcore/health
docker logs --tail 100 goat-lab-v1
docker stop goat-lab-v1
```

The health endpoint must return `ok`.

## Release invariants

- Public packaged artifacts match the hashes in `release/v1_release_manifest.json`.
- The frozen source commit remains `57e504601898afe4e8ead2fa1e51d25990b47de2`.
- The release gate remains 32/32 with zero blockers.
- The production scale remains `bounded_logit_tail`.
- The simulation remains 250,000 runs with seed 23.
- No post-result weight or scaling changes are permitted.
- The tag is created only after local QA, Docker smoke testing, and a clean worktree.
