#!/usr/bin/env bash
# Isolated A/B curriculum launcher (s1→s2→s3) with arbitrary extra Hydra overrides.
#
# Mirrors scripts/train_sequential.py's hot-start threading (each stage resumes the
# previous stage's eval_best) but lets us inject per-arm PPO/gamma overrides that
# train_sequential.py does not pass through. Used for the entropy-vs-gamma A/B.
#
#   ab_curriculum.sh <run_name> [extra hydra overrides...]
#   ab_curriculum.sh abent ppo=aggressive
#   ab_curriculum.sh abgam ++ppo.gamma=0.995      # (RONs must set option_gamma=0.995)
set -euo pipefail

NAME="${1:?usage: ab_curriculum.sh <run_name> [overrides...]}"; shift
EXTRA=("$@")

PROJECT="/home/al-dev-201/projects/algorithm_test_bed"
RL_DIR="$PROJECT/rl"
RUNS="$PROJECT/runs"
export PATH="$RL_DIR/.venv/bin:$PATH"
cd "$RL_DIR"

declare -A TS=( [1]=2500000 [2]=3000000 [3]=3500000 )

PREV=""
for S in 1 2 3; do
  ARGS=( "env=stage$S" "train.run_name=$NAME" "train.total_timesteps=${TS[$S]}" )
  [ -n "$PREV" ] && ARGS+=( "+train.resume=$PREV/eval_best" )
  ARGS+=( "${EXTRA[@]}" )
  echo ">>> [$(date +%H:%M:%S)] stage $S : atb-train ${ARGS[*]}"
  atb-train "${ARGS[@]}"
  # Newest run dir for this stage (epoch-suffixed, fixed-width → lexical = newest).
  PREV=$(ls -d "$RUNS/${NAME}_s${S}_"* | tail -1)
  echo ">>> [$(date +%H:%M:%S)] stage $S done → $PREV"
done
echo ">>> [$(date +%H:%M:%S)] curriculum '$NAME' complete"
