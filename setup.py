from setuptools import setup, find_packages

setup(
    name='VQVAE',
    version='0.1.0',
    url='https://github.com/filipposchiazza/VQVAE',
    author='Filippo Schiazza',
    description='Torch implementation of the Vector Quantized Variational Autoencoder (VQ-VAE) model.',
    packages=find_packages(),    
    install_requires=[
        'torch==1.12.1',
        'numpy==1.25.0',
        'tqdm==4.65.0',
        'matplotlib==3.7.1'
        ],
)