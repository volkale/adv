# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


setup(
    name='advr',
    version='0.0.1',
    description='',
    author='Alexander Volkmann',
    author_email='alexv@gmx.de',
    packages=find_packages(),
    setup_requires=['pytest-runner>=4.2', 'flake8'],
    install_requires=[
        'numpy==1.16.3',
        'xlrd==1.1.0',
        'pandas==0.25.1',  # 0.23.3',
        'pystan==2.18.1.0',
        'scipy==1.2.1',
        'arviz==0.3.3',
        'statsmodels==0.10.1',
        'matplotlib==3.1.0',
        'jupytext==1.2.4',
        'plotly==2.0.8'
    ],
    tests_require=['pytest>=4.0.2', 'pytest-cov>=2.6.0', 'pytest-watch>=4.2.0']
)
