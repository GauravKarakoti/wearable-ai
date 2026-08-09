#!/bin/bash
# Check a built image against the Test Phase submission contract.
#
#   ./validate_image.sh [image-tag]        # default: wearable-ai-eval:latest
#
# Run this before submitting. Organizers run the identical script at intake, so
# a pass here means a pass there -- and a failure caught here costs seconds
# instead of a failed multi-node evaluation run.

# -e included deliberately: without it, an infrastructure failure (no container
# engine, daemon down) would fall through and be reported as a validation
# result. Individual checks are unaffected because they run as `if` conditions,
# which -e does not treat as fatal.
set -euo pipefail

IMAGE="${1:-wearable-ai-eval:latest}"

if command -v podman >/dev/null 2>&1; then
  ENGINE=podman
elif command -v docker >/dev/null 2>&1; then
  ENGINE=docker
else
  echo "Need podman or docker on PATH." >&2
  exit 1
fi

PASS=0
FAIL=0
check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf '  \033[32mPASS\033[0m  %s\n' "$label"; PASS=$((PASS + 1))
  else
    printf '  \033[31mFAIL\033[0m  %s\n' "$label"; FAIL=$((FAIL + 1))
  fi
}
# Validation is the first thing that happens to an untrusted image, and it
# happens on the machine of whoever runs it -- a participant's laptop, or an
# organizer's host at intake. Run it with no network, no capabilities and a
# read-only rootfs: none of the checks below need any of them.
in_image() {
  "$ENGINE" run --rm --entrypoint "" \
    --network=none --cap-drop=ALL --read-only \
    "$IMAGE" "$@"
}

echo "Validating $IMAGE"
echo

echo "Runtime prerequisites"
# The evaluation runtime runs a bash script inside the container before any user
# code. Without bash the job dies with exit 127 and no output.
check "/bin/bash exists (required by the evaluation runtime)" \
      in_image /bin/bash -c 'true'
check "python3 present" in_image /bin/bash -c 'command -v python3'

echo
echo "Python stack"
check "torch imports" in_image /bin/bash -c 'python3 -c "import torch"'
check "torch built with CUDA support" \
      in_image /bin/bash -c 'python3 -c "import torch,sys; sys.exit(0 if torch.version.cuda else 1)"'
check "transformers imports" in_image /bin/bash -c 'python3 -c "import transformers"'
check "opencv imports" in_image /bin/bash -c 'python3 -c "import cv2"'

echo
echo "Starter kit"
check "/app/run_evaluation.py present" in_image /bin/bash -c 'test -f /app/run_evaluation.py'
check "/app/model.py present" in_image /bin/bash -c 'test -f /app/model.py'
check "run_evaluation.py --help works" \
      in_image /bin/bash -c 'cd /app && python3 run_evaluation.py --help'

echo
echo "Model weights"
check "/models exists" in_image /bin/bash -c 'test -d /models'
# Weights must be baked in: compute nodes cannot reach participant storage, and
# downloading at eval time would eat the 300 s/turn budget.
# shellcheck disable=SC2016  # must stay unexpanded -- evaluated inside the container
check "/models is non-empty (weights baked into the image)" \
      in_image /bin/bash -c '[ -n "$(ls -A /models 2>/dev/null)" ]'

echo
# `|| true` is required now that -e is set: an unreadable MODEL_REGISTRY is a
# reportable failure below, not a reason to abort before printing the summary.
MODELS=$(in_image /bin/bash -c 'cd /app && python3 -c "
from model import MODEL_REGISTRY
print(\" \".join(sorted(MODEL_REGISTRY)))
"' 2>/dev/null || true)
if [ -n "$MODELS" ]; then
  echo "Registered model types: $MODELS"
  # Counted as a check so the PASS/FAIL totals below add up to the number of
  # things actually verified.
  PASS=$((PASS + 1))
else
  echo "  (could not read MODEL_REGISTRY -- check model.py imports cleanly)"
  FAIL=$((FAIL + 1))
fi

echo
SIZE=$("$ENGINE" image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null || echo 0)
[ "$SIZE" -gt 0 ] && echo "Image size: $((SIZE / 1000000000)) GB"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "All $PASS checks passed -- image satisfies the submission contract."
  exit 0
fi
echo "$FAIL check(s) failed, $PASS passed. Fix these before submitting."
exit 1
