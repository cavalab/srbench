from setuptools import setup, Extension
setup(name='srbench',
      version='0.0',
      description='Symbolic Regression Benchmarks',
      author='William La Cava, Patryk Orzechowski',
      author_email='williamlacava@gmail.com',
      url='https://github.com/EpistasisLab/regression-benchmarks',
      packages = ['srbench','srbench.methods','srbench.test'],
      package_dir = {'srbench':'experiment'}
)
