#!/usr/bin/env bash
# Verifies the split instruction files (.claude/rules/, .claude/reference/)
# stay internally consistent with .claude/reference/_manifest.json:
#   (a) every section tracked in _manifest.json has a non-empty destination file
#   (b) every .claude/rules/*.md file has valid `paths:` frontmatter
#
# Pure bash + inline python3, zero external deps. Exits non-zero on any
# violation so it can gate CI/pre-push like any other lint check.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MANIFEST=".claude/reference/_manifest.json"
RULES_DIR=".claude/rules"
FAIL=0

if [ ! -f "$MANIFEST" ]; then
  echo "check-instructions: no $MANIFEST found — nothing to check, skipping."
  exit 0
fi

echo "check-instructions: checking manifest destinations are non-empty..."
if ! python3 - "$MANIFEST" <<'PYEOF'
import json
import sys
import os

manifest_path = sys.argv[1]
with open(manifest_path, encoding="utf-8") as f:
    manifest = json.load(f)

fail = False
for section in manifest.get("sections", []):
    heading = section.get("heading", "<unknown heading>")
    dest = section.get("destination")
    if not dest:
        print(f"FAIL: section '{heading}' has no destination in {manifest_path}")
        fail = True
        continue
    if not os.path.isfile(dest):
        print(f"FAIL: section '{heading}' destination '{dest}' does not exist")
        fail = True
        continue
    if os.path.getsize(dest) == 0:
        print(f"FAIL: section '{heading}' destination '{dest}' is empty")
        fail = True

if fail:
    sys.exit(1)
print("OK: every manifest section has a non-empty destination file.")
PYEOF
then
  FAIL=1
fi

echo "check-instructions: checking rule files have valid 'paths:' frontmatter..."
if [ -d "$RULES_DIR" ]; then
  for rule_file in "$RULES_DIR"/*.md; do
    [ -e "$rule_file" ] || continue
    if ! python3 - "$rule_file" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    lines = f.readlines()

if not lines or lines[0].strip() != "---":
    print(f"FAIL: {path} has no '---' frontmatter opening fence")
    sys.exit(1)

closing = None
for i, line in enumerate(lines[1:], start=1):
    if line.strip() == "---":
        closing = i
        break

if closing is None:
    print(f"FAIL: {path} frontmatter never closes with '---'")
    sys.exit(1)

frontmatter = "".join(lines[1:closing])
if "paths:" not in frontmatter:
    print(f"FAIL: {path} frontmatter has no 'paths:' key")
    sys.exit(1)

print(f"OK: {path} has valid paths: frontmatter")
PYEOF
    then
      FAIL=1
    fi
  done
else
  echo "check-instructions: no $RULES_DIR directory, skipping frontmatter check."
fi

if [ "$FAIL" -ne 0 ]; then
  echo "check-instructions: FAILED"
  exit 1
fi

echo "check-instructions: PASSED"
