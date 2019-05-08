from setuptools import setup

MODULES = ['search']

setup(name='spearmint-sklearn',
      description="Sklearn Hyperparameter Optimization based on Spearmint",
      url="//github.com/ZebinYang/spearmint-sklearn.git",
      version='1.0',
      license='GPLv3',
      packages=['spearmint'],
      py_modules=MODULES
     )
