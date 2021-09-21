# -*- coding: utf-8 -*-
from setuptools import setup


def _parse_requirements(file_path):
    lineiter = (line.strip() for line in open(file_path))
    reqs = []
    for line in lineiter:
        reqs.append(line)
    return reqs


install_reqs = _parse_requirements('requirements.txt')

setup(name='pcnaDeep',
      version='1.0',
      description='deep learning pipeline for PCNA-based cell cycle profiling',
      url='https://github.com/Jeff-Gui/PCNAdeep',
      author='Yifan Gui',
      author_email='Yifan.18@intl.zju.edu.cn',
      install_requires=install_reqs,
      license='Apache-2.0',
      packages=['pcnaDeep', 'pcnaDeep.data'],
      zip_safe=False)
