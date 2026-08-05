from __future__ import annotations

import json

from goatlab.reporting.public_release import verify_public_release_bundle


def main() -> int:
    result = verify_public_release_bundle()
    print("GOAT Lab public v1 release verified.")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
