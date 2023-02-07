from pathlib import Path
from setuptools import setup

description = ['Training and evaluation of the 2019 Deep Single Image Camera Calibration paper']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

with open(str(Path(__file__).parent / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

extra_dependencies = ['jupyter', 'scikit-learn', 'ffmpeg-python', 'kornia']

setup(
    name='calib',
    version='1.0',
    packages=['calib'],
    python_requires='>=3.6',
    install_requires=dependencies,
    extras_require={'extra': extra_dependencies},
    author='Alan Savio Paul',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
#     url='https://github.com/cvg/pixloc/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)