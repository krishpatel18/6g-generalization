#!/usr/bin/env bash
# Build a Sionna RT scene for a city box from an offline OpenStreetMap extract.
#
# usage: scripts/build_scene.sh <name> <min_lon> <min_lat> <max_lon> <max_lat> <extract.osm.pbf>
#   e.g. scripts/build_scene.sh wall_street -74.0125 40.7040 -74.0060 40.7085 ~/osm/new-york-latest.osm.pbf
#
# Writes data/scenes/<name>/scene.xml (+ meshes), loadable as source "rt:<name>".
# Extracts: https://download.geofabrik.de (check the .md5 next to each file).
#
# geo2sigmap fetches buildings from the public Overpass API, which refuses
# automated use. This installs geo2sigmap (pinned) into .tools/ and patches it to
# read buildings from a local OSM file cut from the extract with osmium instead.
# Needs: brew install osmium-tool libusb   (libusb is required by open3d)
set -euo pipefail

name=$1; min_lon=$2; min_lat=$3; max_lon=$4; max_lat=$5; extract=$6
root="$(cd "$(dirname "$0")/.." && pwd)"
tools="$root/.tools/geo2sigmap"
commit=08ca0167133ac7de48b9acc3b13a044969543077

if [ ! -x "$tools/.venv/bin/scenegen" ]; then
  mkdir -p "$tools"
  uv venv -q --python 3.11 "$tools/.venv"
  VIRTUAL_ENV="$tools/.venv" uv pip install -q \
    "git+https://github.com/functions-lab/geo2sigmap@${commit}#subdirectory=package"
  "$tools/.venv/bin/python" - <<'EOF'
import pathlib, scene_generation.core as core
p = pathlib.Path(core.__file__); s = p.read_text()
old = '''        buildings = ox.features.features_from_bbox(
            bbox=ground_polygon_4326_bbox, tags={"building": True}
        )'''
new = '''        if os.environ.get("SCENEGEN_OSM_FILE"):
            buildings = ox.features.features_from_xml(
                os.environ["SCENEGEN_OSM_FILE"], tags={"building": True}
            )
        else:
            buildings = ox.features.features_from_bbox(
                bbox=ground_polygon_4326_bbox, tags={"building": True}
            )'''
assert old in s, "geo2sigmap changed; update the patch"
s = s.replace(old, new)
if "\nimport os\n" not in s:
    s = "import os\n" + s
p.write_text(s)
EOF
fi

out="$root/data/scenes/$name"
cut="$(mktemp -d)/$name.osm"
# Pad the cut slightly so buildings straddling the box edge are complete.
osmium extract -b "$(python3 -c "print(f'{$min_lon-0.001},{$min_lat-0.001},{$max_lon+0.001},{$max_lat+0.001}')")" \
  "$extract" -o "$cut" --overwrite
rm -rf "$out"
SCENEGEN_OSM_FILE="$cut" "$tools/.venv/bin/scenegen" bbox "$min_lon" "$min_lat" "$max_lon" "$max_lat" --data-dir "$out"
echo "built $out ($(ls "$out/mesh" | wc -l | tr -d ' ') meshes)"
