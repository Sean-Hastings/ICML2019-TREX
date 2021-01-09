@echo off
setlocal enabledelayedexpansion

for %%x in (%*) do (
python LearnAtariReturn.py --env_name %%x --num_snippets 0 --num_trajs 15000 --grid
:: --resume
python evaluate.py --env_name %%x --model_path learned_models/%%x_s=0_t=15000.params
)
