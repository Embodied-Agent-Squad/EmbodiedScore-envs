#!/usr/bin/env bash
# Every reproduction suite the workspace's legacy runs can feed, one after another.
# Needs EMBODIEDSCORE_DATA_ROOT / EMBODIEDSCORE_SCENE_ROOT and the run dirs below.
set -u
P=${PYTHON:-/home/jian/miniforge3/envs/ac-es/bin/python}
O=${CODING_AGENT_RUNS:-/data/ws_vln/vlnworkspace/outputs/coding-agent}
R=reproduction/reports
D=$(date +%F)
run() { local name=$1; shift; echo "== $name"; $P -u reproduction/replay.py "$@" > "$R/${D}_${name}.log" 2>&1; tail -1 "$R/${D}_${name}.log"; }
run vlnce-r2r_rand100        --benchmark vlnce-r2r --split rand100 --mode vlnce --run $O/codex/std_r2r_codex_gpt-6_default_bare
run vlnce-rxr_rand100        --benchmark vlnce-rxr --split rand100 --mode vlnce --run $O/codex/std_rxr_codex_gpt-6_default_bare
run objectnav-hm3d-v1_mip100 --benchmark objectnav-hm3d-v1 --split mip100 --mode transcript --run $O/claudecode/std_hm3d_cc_opus-5_default_bare
run objectnav-mp3d-v1_mip100 --benchmark objectnav-mp3d-v1 --split mip100 --mode transcript --run $O/claudecode/std_mp3d_cc_opus-5_default_bare
run ovon_mip100_seen          --benchmark ovon --split mip100_seen --mode transcript --run $O/claudecode/std_ovon-seen_cc_opus-5_default_bare
run ovon_mip100_seen_synonyms --benchmark ovon --split mip100_seen_synonyms --mode transcript --run $O/claudecode/std_ovon-syn_cc_opus-5_default_bare
run ovon_mip100_unseen        --benchmark ovon --split mip100_unseen --mode transcript --run $O/claudecode/std_ovon-unseen_cc_opus-5_default_bare
run hmeqa_val500              --benchmark hmeqa --split val --mode actions-log --run $O/claudecode/std_hmeqa500_cc_fable-5_default_bare --eps 0-499
run mthm3d_mip100             --benchmark mthm3d --split mip100 --mode actions-log --run $O/claudecode/std_mthm3d_cc_opus-5_default_bare
run express_mip100            --benchmark express --split mip100 --mode actions-log --run $O/claudecode/std_express_cc_opus-5_default_bare
for f in $R/oracle_goat_*.json; do [ -f "$f" ] && run "goat_$(basename $f .json | sed 's/oracle_goat_//')" --benchmark goat --split $(python3 -c "import json,sys; print(json.load(open('$f'))['split'])") --mode oracle --truth $f; done
for f in $R/oracle_ivlnce_*.json; do [ -f "$f" ] && run "ivlnce_$(basename $f .json | sed 's/oracle_ivlnce_//')" --benchmark ivlnce --split $(python3 -c "import json,sys; print(json.load(open('$f'))['split'])") --mode oracle --truth $f; done
echo "== hmeqa-pose graph log"; $P -u reproduction/pose_log.py --run /data/ws_vln/vlnworkspace/outputs/eval_runs/20260615_183412 > $R/${D}_hmeqa-pose_eval_runs_20260615_183412.log 2>&1; tail -1 $R/${D}_hmeqa-pose_eval_runs_20260615_183412.log
