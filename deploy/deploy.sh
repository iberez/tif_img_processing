#!/usr/bin/env bash
# deploy.sh -- publish a version of tif_img_processing to the shared folder.
#
#   bash deploy/deploy.sh v0.2.0     deploy a tag (recommended) or any ref, e.g. origin/main
#   bash deploy/deploy.sh v0.1.0     an already-built version = instant rollback
#   bash deploy/deploy.sh --list     show deployed versions (* = current)
#
# Maintains, under $ROOT:
#   repo/                     your git clone (the deploy source; colleagues never use it)
#   releases/<version>/       read-only code snapshot, plus env -> ../../envs/<hash>
#   envs/<hash>/              read-only conda env, rebuilt only when environment.yml changes
#   current -> releases/<v>   the version colleagues get
#   bin/microscopy-gui        the one command colleagues run
#
# MICROSCOPY_GROUP=<unix group>  restricts read access to that group instead of
#                                every account on the server.

set -euo pipefail
umask 022

ROOT="${MICROSCOPY_ROOT:-/bigdata/microscopy}"
REPO="$ROOT/repo"
GUI_NOTEBOOK="GUI_tiff_pipeline_cajal.py"
SMOKE_MODULES="tiff_pipeline_pp_functions tiff_pipeline_bridge_functions tiff_pipeline_ss_functions tiff_pipeline_review_functions tiff_pipeline_mega_loop"
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
GROUP="${MICROSCOPY_GROUP:-}"
ME="${USER:-$(id -un)}"

die() { echo "deploy: $*" >&2; exit 1; }

# A failed deploy leaves nothing half-built behind.
PARTIAL=""
trap 'if [[ -n "$PARTIAL" && -d "$PARTIAL" ]]; then rm -rf "$PARTIAL"; fi' EXIT

# Readable by colleagues, writable by nobody -- not even you -- so a stray
# edit or `conda install` can never alter a version people are running.
lock_down() {
    if [[ -n "$GROUP" ]]; then
        chgrp -R "$GROUP" "$1"
        chmod -R u+rX,g+rX,o-rwx,a-w "$1"
    else
        chmod -R a+rX,a-w "$1"
    fi
}

# Top-level folders: colleagues can traverse, you can write. Skips folders you
# don't own (e.g. if an admin created $ROOT for you).
open_dir() {
    [[ -O "$1" ]] || return 0
    if [[ -n "$GROUP" ]]; then
        chgrp "$GROUP" "$1"
        chmod u+rwx,g+rx,o-rwx "$1"
    else
        chmod u+rwx,a+rx "$1"
    fi
}

# Envs are keyed by the hash of environment.yml: unchanged spec -> reuse.
build_env() {
    local spec="$1" dir
    ENV_HASH="$(sha256sum "$spec" | cut -c1-12)"
    dir="$ROOT/envs/$ENV_HASH"
    if [[ -f "$dir/.deploy-ok" ]]; then
        echo "environment $ENV_HASH: unchanged, reusing"
        return
    fi
    if [[ -e "$dir" ]]; then            # leftover from an interrupted build
        chmod -R u+w "$dir"
        rm -rf "$dir"
    fi
    echo "environment $ENV_HASH: building (several minutes the first time)..."
    "$CONDA" env create -p "$dir" -f "$spec"
    # exact record of what got installed, builds and pip packages included
    "$CONDA" env export -p "$dir" > "$ROOT/envs/$ENV_HASH.export.yml"
    touch "$dir/.deploy-ok"
    lock_down "$dir"
}

list_releases() {
    local cur r
    cur="$(readlink "$ROOT/current" 2>/dev/null || true)"
    for r in "$ROOT"/releases/*/; do
        [[ -d "$r" ]] || continue
        r="${r%/}"
        if [[ "releases/${r##*/}" == "$cur" ]]; then printf '* '; else printf '  '; fi
        cat "$r/VERSION" 2>/dev/null || echo "${r##*/}"
    done
}

main() {
    [[ $# -eq 1 ]] || die "usage: deploy.sh <git-ref> | --list"
    if [[ "$1" == "--list" ]]; then list_releases; return; fi
    [[ -n "$CONDA" ]] || die "conda not found -- run this from a shell where 'conda' works"
    [[ -d "$REPO/.git" ]] || die "no git clone at $REPO"

    mkdir -p "$ROOT/releases" "$ROOT/envs" "$ROOT/bin"
    git -C "$REPO" fetch --quiet --tags --force origin \
        || echo "warning: git fetch failed, using refs already in $REPO"

    local sha name dest tmp
    sha="$(git -C "$REPO" rev-parse --verify --quiet "$1^{commit}")" \
        || die "unknown ref '$1' (list tags with: git -C $REPO tag)"
    name="$(git -C "$REPO" describe --tags --always "$sha")"
    name="${name//\//-}"
    dest="$ROOT/releases/$name"

    if [[ -d "$dest" ]]; then
        echo "release $name: already built"
    else
        tmp="$ROOT/releases/.$name.partial"
        PARTIAL="$tmp"
        rm -rf "$tmp"
        mkdir -p "$tmp"
        git -C "$REPO" archive "$sha" | tar -x -C "$tmp"
        [[ -f "$tmp/environment.yml" ]]       || die "$name has no environment.yml at the repo root"
        [[ -f "$tmp/$GUI_NOTEBOOK" ]]         || die "$name has no $GUI_NOTEBOOK"
        [[ -f "$tmp/deploy/microscopy-gui" ]] || die "$name has no deploy/microscopy-gui"

        build_env "$tmp/environment.yml"
        ln -s "../../envs/$ENV_HASH" "$tmp/env"
        chmod 755 "$tmp/deploy/microscopy-gui"
        printf '%s  (commit %s, deployed %s by %s)\n' \
            "$name" "${sha:0:10}" "$(date +%F)" "$ME" > "$tmp/VERSION"

        echo "release $name: smoke-testing imports..."
        if ! ( cd "$tmp" && PYTHONNOUSERSITE=1 PYTHONPATH="$tmp" \
               "$tmp/env/bin/python" -c "import ${SMOKE_MODULES// /, }" ); then
            die "import failed (traceback above) -- fix environment.yml or the code, tag, redeploy"
        fi
        "$tmp/env/bin/python" -m compileall -q "$tmp" > /dev/null

        mv "$tmp" "$dest"
        PARTIAL=""
    fi
    lock_down "$dest"

    # Atomic switch. GUIs already running keep the release they started from;
    # the next launch picks up this one.
    ln -sfn "releases/$name" "$ROOT/.current.new"
    mv -Tf "$ROOT/.current.new" "$ROOT/current"
    ln -sfn "../current/deploy/microscopy-gui" "$ROOT/bin/microscopy-gui"

    local d
    for d in "$ROOT" "$ROOT/bin" "$ROOT/releases" "$ROOT/envs"; do open_dir "$d"; done

    echo
    echo "current -> $(cat "$dest/VERSION")"
    echo "colleague onboarding:  ssh <user>@<server> $ROOT/bin/microscopy-gui --setup"
}

# Everything above is parsed before main runs, so it's safe even if a
# `git pull` rewrites this file mid-run.
main "$@"; exit
