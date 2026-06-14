#!/bin/bash
#
# Submit externally maintained three-pod OPD full-pipeline validation jobs to
# the current kubectl context with a freshly-generated RUN_ID.
#
# Usage:
#   OPD_MANIFEST_DIR=/path/to/manifests scripts/opd/submit_k8s_pipeline.sh [RUN_ID]
#   scripts/opd/submit_k8s_pipeline.sh /path/to/manifests [RUN_ID]
#
# The manifest directory must contain student-sglang.yaml, trainer.yaml, and
# either teacher-xorl.yaml (preferred) or teacher-sglang.yaml. Cluster-specific
# manifests are intentionally kept outside the repo; this helper only rewrites
# CHANGE-ME-RUN-ID and the coord dir.
set -euo pipefail

MANIFEST_DIR="${OPD_MANIFEST_DIR:-}"
if [[ $# -gt 0 && -d "$1" ]]; then
    MANIFEST_DIR="$1"
    shift
fi

if [[ -z "${MANIFEST_DIR}" ]]; then
    echo "error: set OPD_MANIFEST_DIR or pass a manifest directory as the first argument" >&2
    exit 2
fi

RUN_ID="${1:-$(date -u +%Y%m%dt%H%M%Sz)}"
COORD_DIR="/shared/opd-coord/${RUN_ID}"

echo "RUN_ID:    ${RUN_ID}"
echo "Coord dir: ${COORD_DIR}"
echo "Manifests: ${MANIFEST_DIR}"
mkdir -p "${COORD_DIR}"

TEACHER_MANIFEST="teacher-xorl.yaml"
if [[ ! -f "${MANIFEST_DIR}/${TEACHER_MANIFEST}" ]]; then
    TEACHER_MANIFEST="teacher-sglang.yaml"
fi

for manifest in student-sglang.yaml "${TEACHER_MANIFEST}" trainer.yaml; do
    if [[ ! -f "${MANIFEST_DIR}/${manifest}" ]]; then
        echo "error: missing ${MANIFEST_DIR}/${manifest}" >&2
        exit 2
    fi
    echo "==> kubectl apply ${manifest}"
    sed "s#/shared/opd-coord/CHANGE-ME-RUN-ID#${COORD_DIR}#g" "${MANIFEST_DIR}/${manifest}" \
        | sed "s#CHANGE-ME-RUN-ID#${RUN_ID}#g" \
        | kubectl apply -f -
done

echo
echo "Submitted. Tail logs with:"
echo "  kubectl logs -f -l role=trainer"
echo "  kubectl logs -f -l role=student-sglang"
echo "  kubectl logs -f -l role=teacher-xorl"
echo "  kubectl logs -f -l role=teacher-sglang"
echo
echo "Cleanup with:  kubectl delete jobs -l app=opd-pipeline"
