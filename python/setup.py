# Installation script

# setup.py
from setuptools import setup, find_packages

setup(
    name='kfcore',
    version='0.1.0',
    author='Jan Zwiener',
    author_email='jan@zwiener.org',
    description='A lightweight, high-performance Kalman Filter library offering superior numerical stability and efficiency with minimal dependencies.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jnz/KFCore',
    packages=find_packages(),
    install_requires=[
        'numpy','scipy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        # Other classifiers
    ],
)
