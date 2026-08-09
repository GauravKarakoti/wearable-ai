#!/bin/bash
# Script 1 (shared with participants) -- build the submission image.
#
#   ./build_image.sh                      # build, tag as wearable-ai-eval:latest
#   ./build_image.sh --tag my-model:v3    # custom tag
#   ./build_image.sh --push <repo-uri>    # also push to your assigned repository
#   ./build_image.sh --isolation chroot   # if a nested/rootless build cannot mount /proc
#
# --push takes the repository URI issued to your team, complete, e.g.
# <account>.dkr.ecr.us-east-2.amazonaws.com/wearable-ai-2026/<your-team>. Log in
# to it first; this script does not authenticate for you.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The build context is the starter kit directory itself, because that is all the
# Containerfile copies. Keeping it that narrow matters: the context is tarred up
# before the build starts, and the kit can sit inside a repository whose other
# directories hold the dataset.
#
# Three layouts are accepted, so the script works wherever it is dropped: beside
# starter_kit/, one level down with starter_kit/ as a sibling, or in a
# subdirectory of starter_kit/ itself.
if [ -d "$HERE/starter_kit" ]; then
  CONTEXT="$(cd "$HERE/starter_kit" && pwd)"
elif [ -d "$HERE/../starter_kit" ]; then
  CONTEXT="$(cd "$HERE/../starter_kit" && pwd)"
elif [ "$(basename "$(cd "$HERE/.." && pwd)")" = "starter_kit" ]; then
  CONTEXT="$(cd "$HERE/.." && pwd)"
else
  echo "Cannot find the starter kit beside, above, or containing $HERE" >&2
  exit 1
fi

TAG="wearable-ai-eval:latest"
PUSH_TO=""
# Empty means "use the engine default", which is right on an ordinary machine. A
# rootless build nested inside another container cannot mount /proc and fails
# with `Operation not permitted`; chroot isolation skips the namespace that
# fails.
ISOLATION="${PODMAN_ISOLATION:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)       TAG="$2"; shift 2 ;;
    --push)      PUSH_TO="$2"; shift 2 ;;
    --isolation) ISOLATION="$2"; shift 2 ;;
    -h|--help)   sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if command -v podman >/dev/null 2>&1; then
  ENGINE=podman
elif command -v docker >/dev/null 2>&1; then
  ENGINE=docker
else
  echo "Need podman or docker on PATH." >&2
  exit 1
fi

BUILD_ARGS=(build --tag "$TAG" --file "$HERE/Containerfile")
[ -n "$ISOLATION" ] && BUILD_ARGS+=(--isolation "$ISOLATION")

echo "==> Building $TAG with $ENGINE${ISOLATION:+ (isolation=$ISOLATION)}"
"$ENGINE" "${BUILD_ARGS[@]}" "$CONTEXT"

if [ -n "$PUSH_TO" ]; then
  # Only the VERSION of the local tag carries over, never its name. The
  # destination repository is fixed by the organizers -- one per team -- and a
  # team's credentials are pinned to exactly that repository ARN. Appending the
  # local name would address `<repo>/<local-name>`, a different repository that
  # neither exists nor is granted, and ECR answers 403.
  VERSION="${TAG##*:}"
  [ "$VERSION" = "$TAG" ] && VERSION="latest"
  case "${PUSH_TO##*/}" in
    *:*) DEST="$PUSH_TO" ;;          # caller already gave repository:version
    *)   DEST="$PUSH_TO:$VERSION" ;;
  esac

  echo "==> Pushing to $DEST"
  "$ENGINE" tag "$TAG" "$DEST"

  # Submissions are registered and pulled by digest, never by tag -- a tag can
  # be repointed after submission -- so the push has to yield one. The two
  # engines report it differently and only podman has --digestfile; passing that
  # flag to docker fails with an unknown-flag error.
  case "$ENGINE" in
    podman)
      DIGESTFILE="$(mktemp)"
      "$ENGINE" push --digestfile "$DIGESTFILE" "$DEST"
      DIGEST="$(cat "$DIGESTFILE")"
      rm -f "$DIGESTFILE"
      ;;
    docker)
      # docker prints `<version>: digest: sha256:<64 hex> size: <n>` on the way
      # out. tee keeps the participant's normal progress output visible.
      DIGEST="$("$ENGINE" push "$DEST" | tee /dev/stderr \
        | sed -n 's/.*digest: \(sha256:[0-9a-f]\{64\}\).*/\1/p' | tail -1)"
      ;;
  esac

  if [ -z "$DIGEST" ]; then
    echo "Pushed, but could not read back the digest. Find it with:" >&2
    echo "  $ENGINE inspect --format '{{index .RepoDigests 0}}' $DEST" >&2
    exit 1
  fi
  echo "==> Pushed digest: $DIGEST"
  echo "    Register THIS digest with your submission, not the tag."
fi

echo "==> Done."
