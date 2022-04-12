import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "symbolicregression",
    version = "0.0.1",
    author = "Pierre-Alexandre Kamienny",
    author_email = "pakamienny@fb.com",
    description = ("Performing Symbolic Regression with Transformers"),
    license = "BSD",
    keywords = "symbolic regression, transformers",
    url = "",
    packages=find_packages(include=['symbolicregression']),
    long_description=read('README.md'),
)
