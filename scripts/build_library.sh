#!/usr/bin/env bash
# Build every scene in scripts/scene_library.txt from one OSM extract.
# usage: scripts/build_library.sh <extract.osm.pbf>
set -euo pipefail
cd "$(dirname "$0")/.."
grep -v '^#' scripts/scene_library.txt | while read -r name a b c d _; do
  [ -f "data/scenes/$name/scene.xml" ] && { echo "have $name"; continue; }
  scripts/build_scene.sh "$name" "$a" "$b" "$c" "$d" "$1" 2>&1 | grep -E "^built|Error|Traceback" || true
done
