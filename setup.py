import setuptools
from setuptools import setup

setup(name='gym_RLcourse',
      version='0.0.1',
      author='Per Mattsson',
      author_email='magni84@gmail.com',
      description='Some environments used in the RL course at Uppsala University',
      url='https://github.com/magni84/gym_RLcourse',
      packages=setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
      install_requires=['gymnasium', 'numpy']
)
