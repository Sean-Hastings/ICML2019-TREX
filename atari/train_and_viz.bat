@echo off
setlocal enabledelayedexpansion

for %%x in (%*) do (
python LearnAtariReturn.py --env_name %%x --num_snippets 0 --num_trajs 3000
python evaluate.py --env_name %%x --model_path learned_models/%%x_s=0_t=3000.params
)
