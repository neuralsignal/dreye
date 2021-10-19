#!/usr/bin/env python

import setuptools
import os

path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(path, 'requirements.txt')) as f:
    requirements = f.read().split()

# User README.md as long description
with open("README.md", encoding="utf-8") as f:
    README = f.read()


setuptools.setup(
    name='dreye',
    version='0.3.0dev1',
    description='Dreye: Color models and stimuli for all model organisms',
    long_description=README,
    author='Matthias Christenson',
    author_email='gucky@gucky.eu',
    packages=setuptools.find_packages(exclude=['tests', 'docs']),
    install_requires=requirements, 
    include_package_data=True,
)
