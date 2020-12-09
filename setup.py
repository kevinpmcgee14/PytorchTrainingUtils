from setuptools import setup, find_packages

setup(
    name='training_utils', 
    version='1.0', 
    packages=find_packages('training_utils', 'training_utils.*'),
    install_requires=[
        'pandas==1.1.4',
        'torch==1.7.0+cu101',
        'torchvision==0.8.1+cu101',
        'tqdm==4.41.1',
        'torch-lr-finder==0.2.1'
    ],

    )