#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec python "${REPO_ROOT}/scripts/opd/run_opd_pipeline.py" --help
fi

: "${XORL_TRAIN_URL:?Set XORL_TRAIN_URL to the student trainer API URL.}"
: "${OPD_COORD_DIR:?Set OPD_COORD_DIR to the shared coord dir containing student.json and teacher.json.}"
: "${OPD_TEACHER_HEAD:?Set OPD_TEACHER_HEAD to the AR teacher LM-head source.}"

OPD_MTP_K_TOKS="${OPD_MTP_K_TOKS:-2}"
OPD_MTP_MASK_TOKEN_ID="${OPD_MTP_MASK_TOKEN_ID:-248063}"
OPD_MTP_CONF_THRESHOLD="${OPD_MTP_CONF_THRESHOLD:-0.6}"

export OPD_TEACHER_BACKEND="${OPD_TEACHER_BACKEND:-xorl}"
export OPD_NUM_STEPS="${OPD_NUM_STEPS:-1}"
export OPD_MAX_NEW_TOKENS="${OPD_MAX_NEW_TOKENS:-8}"
export OPD_SKIP_OPTIM_STEP="${OPD_SKIP_OPTIM_STEP:-1}"
export OPD_PROMPTS_JSON="${OPD_PROMPTS_JSON:-[[248045,846,198,3710,369,220,17,10,17,30,248046,198,248045,74455,198]]}"
export OPD_STUDENT_SAMPLING_PARAMS_JSON="${OPD_STUDENT_SAMPLING_PARAMS_JSON:-{\"top_k\":1}}"
export OPD_SINGLESHOT_MTP_JSON="${OPD_SINGLESHOT_MTP_JSON:-{\"k_toks\":${OPD_MTP_K_TOKS},\"mask_token_id\":${OPD_MTP_MASK_TOKEN_ID},\"attention_bias_dtype\":\"bfloat16\",\"sampling_mode\":\"native\",\"train_rollout_strategy\":\"confidence\",\"mtp_strategy\":[\"conf_adapt\",${OPD_MTP_CONF_THRESHOLD}],\"mtp_conf_threshold\":${OPD_MTP_CONF_THRESHOLD},\"rollout_replay\":true,\"validate_native_mtp_trace\":true,\"native_mtp_debug_trace\":true}}"

exec python "${REPO_ROOT}/scripts/opd/run_opd_pipeline.py" "$@"
