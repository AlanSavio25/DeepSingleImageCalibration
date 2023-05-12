from pathlib import Path

root = Path(__file__).parent #.parent  # top-level directory
# DATA_PATH = root / 'calib' / 'minidata/'
DATA_PATH = '/cluster/project/infk/cvg/students/alpaul/DeepSingleImageCalibration/'
DATASETS_PATH = root / 'calib' / 'datasets/'  # datasets and pretrained weights
# TRAINING_PATH = '/cluster/project/infk/cvg/students/alpaul/DeepSingleImageCalibration/outputs/training/'  # training checkpoints
TRAINING_PATH = '/cluster/scratch/alpaul'  # training checkpoints
EVAL_PATH = '/cluster/project/infk/cvg/students/alpaul/DeepSingleImageCalibration/outputs/results/'  # evaluation results
