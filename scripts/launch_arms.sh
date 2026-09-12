#!/bin/bash
# usage: launch_arms.sh SEEDS config1.yaml [config2.yaml ...]  -- one detached process per arm; all its seeds train together
export PATH=$HOME/.local/bin:$PATH; cd /root/fsf/repo; mkdir -p /root/fsf/logs
seeds=$1; shift
for cfg in "$@"; do
  name=$(basename $cfg .yaml)
  setsid nohup uv run python -m fsf.train $cfg --seeds $seeds > /root/fsf/logs/$name.log 2>&1 < /dev/null &
  echo "launched $cfg seeds=$seeds pid=$!"
done
