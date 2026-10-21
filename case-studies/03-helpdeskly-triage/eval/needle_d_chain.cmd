@echo off
rem Candidate D (Needle 3 LoRA classifier) on CPU, beside the GPU queue.
rem 1) both archives on val, 2) the one with the higher val intent macro-F1 on test, 3) base on test.
cd /d %~dp0
set PY=..\.venv\Scripts\python.exe
%PY% predict_d_needle.py --stage after --weights ../train/needle_out/tuned.cact --split val --tag d_needle_val_default > ..\results\run_d_val_default.log 2>&1
%PY% predict_d_needle.py --stage after --weights ../train/needle_out/tuned_fullkv.cact --split val --tag d_needle_val_fullkv > ..\results\run_d_val_fullkv.log 2>&1
for /f %%w in ('%PY% -c "import json;f=lambda t:json.load(open('../artifacts/preds_d_needle_val_'+t+'.json'))['metrics']['intent']['macro_f1'];print('tuned_fullkv.cact' if f('fullkv')>=f('default') else 'tuned.cact')"') do set BEST=%%w
echo chosen %BEST% > ..\results\run_d_choice.log
%PY% predict_d_needle.py --stage after --weights ../train/needle_out/%BEST% > ..\results\run_d_after.log 2>&1
%PY% predict_d_needle.py --stage before > ..\results\run_d_before.log 2>&1
echo done > ..\results\needle_d_chain.done
