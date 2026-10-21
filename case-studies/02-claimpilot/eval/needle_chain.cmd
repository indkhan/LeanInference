@echo off
rem Case 2 Needle evals on CPU (Windows engine), beside the GPU queue. Each step resumes from its cache.
cd /d %~dp0..
set PY=.venv\Scripts\python.exe
%PY% eval\needle_extract.py --label needle3-ft --weights train/needle_out/tuned.cact --tag val --split val > eval\logs\needle_ft_val.log 2>&1
%PY% eval\needle_extract.py --label needle3-ft-fullkv --weights train/needle_out/tuned_fullkv.cact --tag val --split val > eval\logs\needle_ft_fullkv_val.log 2>&1
%PY% eval\needle_extract.py --label needle3-ft-fullkv --weights train/needle_out/tuned_fullkv.cact --tag after --split test > eval\logs\needle_ft_fullkv_test.log 2>&1
%PY% eval\needle_extract.py --label needle3-base --tag before --split test > eval\logs\needle_base_test.log 2>&1
echo done > eval\logs\needle_chain.done
