@echo off
setlocal enabledelayedexpansion

for %%x in (%*) do (
python LearnAtariReward.py --env_name %%x --num_snippets 0 --num_trajs 3000
python VisualizeAtariLearnedReward.py --env_name %%x
python utils/viz/quickviz.py --env_name %%x
)

python utils/viz/plot_correlation.py
