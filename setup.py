# -*- coding: utf-8 -*-

from distutils.core import setup
import shutil
shutil.copy('README.md', 'pynumwrap/README.md')

setup(name='pynumwrap',
      version='0.16',
      description='Python package wrapping python and mpmath types behind a common interface.',
      author="Peter Bingham",
      author_email="petersbingham@hotmail.co.uk",
      packages=['pynumwrap'],
      package_data={'pynumwrap': ['README.md']}
     )
