#!/bin/bash
export WANDB_PROJECT=bbheron
export PROJECT_NAME=my_vlm/exp001
export WANDB_NAME=$PROJECT_NAME

deepspeed train.py --config_file projects/$PROJECT_NAME.yml
