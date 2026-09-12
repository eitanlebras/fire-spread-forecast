#!/bin/bash
# One-shot pod bootstrap on local disk (no network volume). Idempotent. Logs to /root/fsf/logs/bootstrap.log
set -x; export PATH=$HOME/.local/bin:$PATH
mkdir -p /root/fsf/logs /root/data/ndws /root/data/wfts /root/fsf/results
apt-get install -y -qq unzip >/dev/null 2>&1
which uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d /root/fsf/repo ] || git clone -q https://github.com/eitanlebras/fire-spread-forecast.git /root/fsf/repo
cd /root/fsf/repo && git pull -q && uv sync -q --python 3.12
[ -d /root/WildfireSpreadTS_code ] || git clone -q https://github.com/SebastianGer/WildfireSpreadTS.git /root/WildfireSpreadTS_code
# NDWS (HF mirror) + fp16 caches, in background
setsid nohup bash -c "export PATH=\$HOME/.local/bin:\$PATH; cd /root/fsf/repo && uv run --with huggingface_hub --with hf_xet python -c \"from huggingface_hub import snapshot_download; snapshot_download('TheRootOf3/next-day-wildfire-spread', repo_type='dataset', local_dir='/root/data/ndws')\" && uv run python -m fsf.data /root/data/ndws/data ndws12 ndws12_raw_th derived; echo NDWS_DONE" > /root/fsf/logs/ndws.log 2>&1 < /dev/null &
# WFTS fires via HTTP ranges: all 2018+2019 (train), seed-0 sample of 30 for 2020 (val) and 2021 (test); 8 shards each
for i in $(seq 0 7); do
  setsid nohup .venv/bin/python -m fsf.wfts_remote fetch /root/data/wfts/tif 2018 2019 --shard $i/8 > /root/fsf/logs/fetch_train_$i.log 2>&1 < /dev/null &
  setsid nohup .venv/bin/python -m fsf.wfts_remote fetch /root/data/wfts/tif 2020 2021 --n 30 --shard $i/8 > /root/fsf/logs/fetch_evaltest_$i.log 2>&1 < /dev/null &
done
# OlmoEarth minimal + cu128 torch, in background
setsid nohup bash -c "export PATH=\$HOME/.local/bin:\$PATH; cd /root && git clone -q https://github.com/allenai/olmoearth_pretrain_minimal.git && cd olmoearth_pretrain_minimal && uv sync -q && uv pip install -q --python .venv/bin/python --reinstall 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128 && echo OLMO_DONE" > /root/fsf/logs/olmoearth_setup.log 2>&1 < /dev/null &
echo BOOTSTRAP_LAUNCHED
