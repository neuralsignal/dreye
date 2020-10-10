#!/usr/bin/env python

import setuptools
import os

path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(path, 'requirements.txt')) as f:
    requirements = f.read().split()

setuptools.setup(
    name='dreye',
    version='0.0.3',
    description='Dreye: Color models and stimuli for all model organisms',
    author='Matthias Christenson',
    author_email='gucky@gucky.eu',
    # install_requires=['requirements.txt'],
    # TODO requirement file
    packages=setuptools.find_packages(exclude=['tests', 'docs']),
    install_requires=requirements
)
