#!/bin/bash
# One-shot pod bootstrap on local disk, plain pip (uv deadlocked on .venv/.lock). Idempotent.
set -x
mkdir -p /root/fsf/logs /root/data/ndws /root/data/wfts/tif /root/fsf/results
apt-get install -y -qq unzip >/dev/null 2>&1
[ -d /root/fsf/repo ] || git clone -q https://github.com/eitanlebras/fire-spread-forecast.git /root/fsf/repo
cd /root/fsf/repo && git pull -q
python3 -c "import requests" 2>/dev/null || pip install -q requests
# 1. WFTS fires via HTTP ranges (critical path): all 2018+2019, seed-0 sample of 30 for 2020 and 2021; 8 shards each
for i in $(seq 0 7); do
  setsid nohup python3 -m fsf.wfts_remote fetch /root/data/wfts/tif 2018 2019 --shard $i/8 > /root/fsf/logs/fetch_train_$i.log 2>&1 < /dev/null &
  setsid nohup python3 -m fsf.wfts_remote fetch /root/data/wfts/tif 2020 2021 --n 30 --shard $i/8 > /root/fsf/logs/fetch_evaltest_$i.log 2>&1 < /dev/null &
done
# 2. training venv with pip
setsid nohup bash -c "python3 -m venv /root/fsf/venv && /root/fsf/venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 && /root/fsf/venv/bin/pip install -q numpy xarray 'zarr<3' pyarrow pandas scikit-learn pyyaml tqdm rasterio pystac-client requests huggingface_hub hf_xet && echo VENV_DONE" > /root/fsf/logs/venv.log 2>&1 < /dev/null &
# 3. NDWS from HF mirror (cache build waits for the venv)
setsid nohup bash -c "pip install -q huggingface_hub hf_xet && python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('TheRootOf3/next-day-wildfire-spread', repo_type='dataset', local_dir='/root/data/ndws')\" && echo NDWS_DL_DONE" > /root/fsf/logs/ndws.log 2>&1 < /dev/null &
# 4. OlmoEarth minimal in its own venv
setsid nohup bash -c "cd /root && git clone -q https://github.com/allenai/olmoearth_pretrain_minimal.git && cd olmoearth_pretrain_minimal && python3 -m venv .venv && .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu128 && .venv/bin/pip install -q -e . && echo OLMO_DONE" > /root/fsf/logs/olmoearth_setup.log 2>&1 < /dev/null &
[ -d /root/WildfireSpreadTS_code ] || git clone -q https://github.com/SebastianGer/WildfireSpreadTS.git /root/WildfireSpreadTS_code
echo BOOTSTRAP_LAUNCHED
