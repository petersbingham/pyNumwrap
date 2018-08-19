# -*- coding: utf-8 -*-

from distutils.core import setup
import os
import shutil
shutil.copy('README.md', 'pynumwrap/README.md')

dir_setup = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_setup, 'pynumwrap', 'release.py')) as f:
    # Defines __version__
    exec(f.read())

setup(name='pynumwrap',
      version=__version__,
      description='Python package wrapping python and mpmath types behind a common interface.',
      author="Peter Bingham",
      author_email="petersbingham@hotmail.co.uk",
      packages=['pynumwrap'],
      package_data={'pynumwrap': ['README.md']}
     )
