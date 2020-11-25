@echo off
setlocal enabledelayedexpansion

for %%x in (%*) do (
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/tflogs python -m baselines.run --alg=ppo2 --env=%%x --seed 0 --num_timesteps=2500  --save_interval=25
)

:: THIS HAS NOT BEEN TESTED OR USED
