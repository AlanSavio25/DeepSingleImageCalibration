from pathlib import Path

root = Path(__file__).parent #.parent  # top-level directory
#DATA_PATH = root / 'calib' / 'minidata/'
DATA_PATH = ''
DATASETS_PATH = root / 'calib' / 'datasets/'  # datasets and pretrained weights
TRAINING_PATH = ''  # training checkpoints
EVAL_PATH = ''  # evaluation results
