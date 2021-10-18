from setuptools import setup, find_packages

setup(
    name='training_utils', 
    version='0.0.2', 
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torch-lr-finder',
        'pandas==1.1.4',
        'tqdm==4.48.0'
    ],

    )
