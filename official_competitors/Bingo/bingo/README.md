![Bingo Logo](media/logo.png)

master: [![Build Status](https://travis-ci.com/nasa/bingo.svg?branch=master)](https://travis-ci.com/nasa/bingo) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingo/badge.svg?branch=master)](https://coveralls.io/github/nasa/bingo?branch=master)

develop: 
[![Build Status](https://github.com/nasa/bingo/actions/workflows/tests.yml/badge.svg?branch=develop)](https://github.com/nasa/bingo/actions?query=branch%3Adevelop)
[![Coverage Status](https://coveralls.io/repos/github/nasa/bingo/badge.svg?branch=develop)](https://coveralls.io/github/nasa/bingo?branch=develop) 
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9fe09cffafe64032962a82f7f1588e9f)](https://www.codacy.com/app/bingo_developers/bingo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nasa/bingo&amp;utm_campaign=Badge_Grade)

## General
Bingo is an open source package for performing symbolic regression, Though it 
can be used as a general purpose evolutionary optimization package.  

### Key Features
*   Integrated local optimization strategies
*   Parallel island evolution strategy implemented with mpi4py
*   Coevolution of fitness predictors
  
### Note
At this point, the API is still in a state of flux. The current release has a 
much more stable API but still lacks some of the features of older releases.

## Getting Started

### Cloning with git
The Bingo repository uses git submodules so make sure to clone all the
submodules when cloning.  Git has an easy way to do this with:
```shell
git clone --recurse-submodules ...
```

### Dependencies
Bingo is intended for use with Python 3.x.  Bingo requires installation of a 
few dependencies which are relatively common for data science work in python:
*   numpy
*   scipy
*   matplotlib
*   mpi4py (if parallel implementations are to be run)
*   pytest, pytest-mock (if the testing suite is to be run)
  
A `requirements.txt` file is included for easy installation of dependencies with 
`pip` or `conda`.

Installation with pip:
```shell
pip install -r requirements.txt
```

Installation with conda:
```shell
conda install --yes --file requirements.txt
```

### BingoCpp
A section of bingo is written in c++ for increased performance.  In order to 
take advantage of this capability, the code must be compiled.  See the 
documentation in the bingocpp submodule for more information.

Note that bingo can be run without the bingocpp portion, it will just have lower 
performance.

If bingocpp has been properly installed, the following command should run 
without error.
```shell
python -c "from bingocpp"
```

A common error in the installation of bingocpp is that it must be built with 
the same version of python that will run your bingo scripts.  The easiest way 
to ensure consistent python versioning is to build and run in a Python 3 
virtual environment.

### Documentation
Sphynx is used for automatically generating API documentation for bingo. The 
most recent build of the documentation can be found in the repository at: 
`doc/_build/html/index.html`

## Running Tests
An extensive unit test suite is included with bingo to help ensure proper 
installation. The tests can be run using pytest on the tests directory, e.g., 
by running:
```shell
python -m pytest tests 
```
from the root directory of the repository.

## Usage Examples
The best place to get started in bingo is by going through the jupyter notebook
tutorials in the [examples directory](examples/). They step through several of
the most important aspects of running bingo with detailed explanations at each
step.  The [examples directory](examples/) also contains several python scripts
which may act as a good base when starting to write your own custom bingo
scripts.

## Contributing
1.  Fork it (<https://github.com/nasa/bingo/fork>)
2.  Create your feature branch (`git checkout -b feature/fooBar`)
3.  Commit your changes (`git commit -am 'Add some fooBar'`)
4.  Push to the branch (`git push origin feature/fooBar`)
5.  Create a new Pull Request

## Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available, 
see the [tags on this repository](https://github.com/nasa/bingo/tags). 

## Authors
*   Geoffrey Bomarito
*   Tyler Townsend
*   Jacob Hochhalter
*   Ethan Adams
*   Kathryn Esham
*   Diana Vera
  
## License 
Copyright 2018 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. No copyright is claimed in 
the United States under Title 17, U.S. Code. All Other Rights Reserved.

The Bingo Mini-app framework is licensed under the Apache License, Version 2.0 
(the "License"); you may not use this application except in compliance with the 
License. You may obtain a copy of the License at 
http://www.apache.org/licenses/LICENSE-2.0 .

Unless required by applicable law or agreed to in writing, software distributed 
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied. See the License for the 
specific language governing permissions and limitations under the License.
