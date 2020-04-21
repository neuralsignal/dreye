#!/usr/bin/env python

from setuptools import setup

setup(
    name='dreye',
    version='0.1',
    description='Dreye: Color models and stimuli for all model organisms',
    author='Matthias Christenson',
    author_email='gucky@gucky.eu',
    install_requires=['requirements.txt'],
    # TODO requirement file
    packages=['dreye']
)
