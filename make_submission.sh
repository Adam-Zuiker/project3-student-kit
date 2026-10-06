#!/usr/bin/env bash
# Packages a completed starter/<lang> directory into submission.tar.gz, with
# that directory's own Makefile sitting at the archive root -- the shape
# the Project 3 handout's (docs/PROJECT3_*.md) submission section
# requires. Run this from the kit root: ./make_submission.sh <c|cpp|python|rust>
set -euo pipefail

kit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
languages=(c cpp python rust)

usage() {
    local joined
    joined="$(IFS='|'; echo "${languages[*]}")"
    echo "usage: $(basename "${BASH_SOURCE[0]}") <$joined>" >&2
    exit 2
}

lang="${1:-}"
[[ -n "$lang" ]] || usage
match=0
for candidate in "${languages[@]}"; do
    [[ "$candidate" == "$lang" ]] && match=1
done
(( match )) || usage

src_dir="$kit_root/starter/$lang"
if [[ ! -f "$src_dir/Makefile" ]]; then
    echo "error: $src_dir has no Makefile -- is the starter kit intact?" >&2
    exit 1
fi

output="$kit_root/submission.tar.gz"

# List top-level entries by name (rather than tarring "."): GNU tar includes
# an explicit "." directory member for `tar -C dir -czf out .`, which the
# autograder's archive-safety check rejects outright
# as an unsafe path. Named entries avoid that member entirely.
entries=()
while IFS= read -r entry; do
    entries+=("$entry")
done < <(
    shopt -s dotglob nullglob
    for path in "$src_dir"/*; do
        printf '%s\n' "$(basename "$path")"
    done | sort
)

# COPYFILE_DISABLE stops macOS bsdtar from adding AppleDouble `._*`
# companions for files with extended attributes (e.g. quarantine flags from
# an unzipped download); on Linux those become bogus `._foo.py` sources that
# break `python3 -m compileall`.
COPYFILE_DISABLE=1 tar --exclude-vcs \
    --exclude='build' --exclude='bin' --exclude='.build' --exclude='target' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
    --exclude='.DS_Store' --exclude='._*' \
    -C "$src_dir" -czf "$output" "${entries[@]}"

# Sanity-check the shape we just produced before handing it to the student:
# the autograder requires a Makefile at the archive root (or inside a
# single wrapping directory, which we don't produce here).
# (List into a variable first: `tar -t | grep -q` under pipefail fails
# spuriously on a large archive, because grep -q exits early and tar dies of
# SIGPIPE -- e.g. the Rust starter with its vendored crate.)
listing="$(tar -tzf "$output" | sed 's#^\./##')"
if ! grep -qx 'Makefile' <<<"$listing"; then
    echo "error: produced archive has no top-level Makefile -- this is a bug in $(basename "${BASH_SOURCE[0]}")" >&2
    rm -f "$output"
    exit 1
fi

echo "wrote $output ($(du -h "$output" | cut -f1)) from starter/$lang"
