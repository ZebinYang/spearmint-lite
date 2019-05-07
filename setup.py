from setuptools import setup

MODULES = ['ExperimentGrid', 'gp', 'helpers', 'Locker', 'sobol_lib', 'util']

setup(name='spearmint',
      description="Practical Bayesian Optimization of Machine Learning Algorithms",
      author="Jasper Snoek, Hugo Larochelle, Ryan P. Adams",
      url="https://github.com/JasperSnoek/spearmint",
      version='1.0',
      license='GPLv3',
      packages=['spearmint'],
      py_modules=MODULES
     )
