from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ASSET_DIR = Path("release/assets")
MANIFEST_PATH = Path("release/v1_release_manifest.json")


def _write_or_verify(path: Path, content: str, verify: bool) -> None:
    if verify:
        if not path.exists():
            raise FileNotFoundError(path)
        actual = path.read_text(encoding="utf-8")
        if actual != content:
            raise ValueError(f"Release asset is stale: {path}")
        print("VERIFIED:", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print("WROTE:", path)


def _hero_svg(manifest: dict[str, object]) -> str:
    central = manifest["central_result"]
    simulation = manifest["simulation_result"]
    players = {
        row["player"]: row
        for row in simulation["players"]
    }
    margin = float(central["margin_points"])
    lebron_rate = float(players["LeBron James"]["win_rate"])

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
  <rect width="1600" height="900" fill="#080b12"/>
  <rect x="70" y="70" width="1460" height="760" rx="36" fill="#111827" stroke="#f59e0b" stroke-width="3"/>
  <text x="130" y="170" fill="#f8fafc" font-family="Arial, sans-serif" font-size="64" font-weight="700">GOAT Lab v1</text>
  <text x="130" y="235" fill="#94a3b8" font-family="Arial, sans-serif" font-size="30">Michael Jordan vs. LeBron James — frozen 2025–26 release</text>
  <text x="130" y="350" fill="#f59e0b" font-family="Arial, sans-serif" font-size="54" font-weight="700">LeBron ranks first by {margin:.6f} points</text>
  <text x="130" y="425" fill="#f8fafc" font-family="Arial, sans-serif" font-size="38">LeBron wins {lebron_rate:.4%} of 250,000 hierarchy-aware weight systems</text>
  <rect x="130" y="500" width="900" height="54" rx="27" fill="#1f2937"/>
  <rect x="130" y="500" width="{900 * lebron_rate:.1f}" height="54" rx="27" fill="#f59e0b"/>
  <text x="130" y="625" fill="#f8fafc" font-family="Arial, sans-serif" font-size="31">The result is conditional, not an objective probability.</text>
  <text x="130" y="680" fill="#94a3b8" font-family="Arial, sans-serif" font-size="27">Approved scaling scenarios split 2–2 between the players.</text>
  <text x="130" y="735" fill="#94a3b8" font-family="Arial, sans-serif" font-size="27">Defense is the largest Jordan swing factor; offense is LeBron’s largest counterweight.</text>
</svg>
"""


def _scale_svg(manifest: dict[str, object]) -> str:
    scenarios = manifest["scale_sensitivity"]["scenarios"]
    rows = []
    max_abs = max(abs(float(row["lebron_minus_jordan"])) for row in scenarios)
    for index, row in enumerate(scenarios):
        y = 180 + index * 150
        margin = float(row["lebron_minus_jordan"])
        width = 480 * abs(margin) / max_abs
        x = 800 if margin >= 0 else 800 - width
        winner = html.escape(str(row["winner"]))
        scenario = html.escape(str(row["scenario"]))
        fill = "#f59e0b" if margin >= 0 else "#cbd5e1"
        rows.append(
            f'<text x="120" y="{y + 16}" fill="#f8fafc" font-family="Arial, sans-serif" '
            f'font-size="28">{scenario}</text>'
            f'<rect x="{x:.1f}" y="{y - 24}" width="{width:.1f}" height="48" rx="12" fill="{fill}"/>'
            f'<text x="1340" y="{y + 16}" text-anchor="end" fill="#f8fafc" '
            f'font-family="Arial, sans-serif" font-size="26">{margin:+.3f} — {winner}</text>'
        )

    joined = "\n  ".join(rows)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
  <rect width="1600" height="900" fill="#080b12"/>
  <text x="120" y="90" fill="#f8fafc" font-family="Arial, sans-serif" font-size="52" font-weight="700">The winner changes with the scaling method</text>
  <text x="120" y="135" fill="#94a3b8" font-family="Arial, sans-serif" font-size="26">LeBron − Jordan score margin; positive is LeBron, negative is Jordan</text>
  <line x1="800" y1="155" x2="800" y2="775" stroke="#64748b" stroke-width="3"/>
  {joined}
  <text x="120" y="840" fill="#94a3b8" font-family="Arial, sans-serif" font-size="27">That is why GOAT Lab labels the v1 conclusion conditional rather than universal.</text>
</svg>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = {
        ASSET_DIR / "goat_lab_v1_result.svg": _hero_svg(manifest),
        ASSET_DIR / "scale_sensitivity.svg": _scale_svg(manifest),
    }

    for path, content in assets.items():
        _write_or_verify(path, content, args.verify)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
