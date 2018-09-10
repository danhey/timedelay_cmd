#!/usr/bin/env python

import os
import sys
from setuptools import setup

entry_points = {'console_scripts': [
        'timedelay = timedelay.timedelay:timedelay_main',
]}

setup(name='timedelay',
      version=0.1,
      packages=['timedelay'],
      install_requires=['numpy>=1.11', 'astropy>=1.3', 'lightkurve',
                        'matplotlib>=1.5.3'],
      entry_points=entry_points,
      include_package_data=True,
    )