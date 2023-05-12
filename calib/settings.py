from pathlib import Path

root = Path(__file__).parent #.parent  # top-level directory
#DATA_PATH = root / 'calib' / 'minidata/'
DATA_PATH = 'path/to/folder/containing/{data_dir}' # data_dir should be set in calib/calib/configs/config_train.py. data_dir folder should contain *.json, *.jpg.
DATASETS_PATH = root / 'calib' / 'datasets/'  # datasets and pretrained weights
TRAINING_PATH = ''  # training checkpoints. This is where the training logs and checkpoints are stored.
EVAL_PATH = ''  # evaluation results
