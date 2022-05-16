import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), 
                       open(project + '/__init__.py').read())
    return result.group(1)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                         out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'bingocpp'] 
                              + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="bingo-nasa",
    version=get_property('__version__', 'bingo'),
    author="Geoffrey Bomarito",
    author_email="geoffrey.f.bomarito@nasa.gov",
    description="A package for genetic optimization and symbolic regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/bingo",
    packages=["bingo",
              "bingo.chromosomes",
              "bingo.evaluation",
              "bingo.evolutionary_algorithms",
              "bingo.evolutionary_optimizers",
              "bingo.local_optimizers",
              "bingo.selection",
              "bingo.stats",
              "bingo.symbolic_regression",
              "bingo.symbolic_regression.agraph",
              "bingo.symbolic_regression.agraph.evaluation_backend",
              "bingo.symbolic_regression.agraph.simplification_backend",
              "bingo.symbolic_regression.benchmarking",
              "bingo.util",
              "bingo.variation",
              "bingocpp"
             ],
    install_requires=['mpi4py',
                      'numpy',
                      'scipy',
                      'dill'],
    python_requires='~=3.4',
    classifiers=[
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha"
    ],
    # add extension module
    ext_modules=[CMakeExtension('bingo.bingocpp', 'bingocpp')],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
