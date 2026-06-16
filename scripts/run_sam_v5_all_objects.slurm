#!/bin/bash
#SBATCH --job-name=sam_v5_boxes
#SBATCH --account=drjieliu99
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/turbo/umms-drjieliu/usr/mxzhu/Model/My/rule_discovery/sam_v5_all_objects_trial/debug/sam_v5_full_run_%j.log

set -euo pipefail

cd /nfs/turbo/umms-drjieliu/usr/mxzhu

export SAM_RESUME_EXISTING=1
export SAM_CHECKPOINT_PATH=/nfs/turbo/umms-drjieliu/usr/mxzhu/Model/My/checkpoints/sam_vit_b_01ec64.pth

Model/.venv/bin/python Model/My/scripts/32_run_sam_boxes_v5_all_objects.py
