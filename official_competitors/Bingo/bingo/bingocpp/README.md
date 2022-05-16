# BingoCpp #

![Bingo Logo](media/logo.png)

master: [![Build Status](https://travis-ci.com/nasa/bingocpp.svg?branch=master)](https://travis-ci.com/nasa/bingocpp) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingocpp/badge.svg?branch=master)](https://coveralls.io/github/nasa/bingocpp?branch=master)

develop: [![Build Status](https://travis-ci.com/nasa/bingocpp.svg?branch=develop)](https://travis-ci.com/nasa/bingocpp) [![Coverage Status](https://coveralls.io/repos/github/nasa/bingocpp/badge.svg?branch=develop)](https://coveralls.io/github/nasa/bingocpp?branch=develop) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/ccd11c4092544eaca355722cea87272e)](https://www.codacy.com/app/bingo_developers/bingocpp?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nasa/bingocpp&amp;utm_campaign=Badge_Grade)

## General ##

BingoCpp is part of the open source package Bingo for performing symbolic
regression.  BingoCpp contains the c++ implementation of a portion of the code
within bingo. 

## Getting Started ##

### Cloning ###

BingoCpp has 3 submodules: eigen, google test, and pybind.  To clone this
repository and include the submodules, run the following command:

```bash
git clone --recurse-submodules https://github.com/nasa/bingocpp
```

### Installation/Compiling with CMake ###

Installing from source requires git and a recent version of
[cmake](https://cmake.org/).

Installation can be performed using the typical out-of-source build flow:

```bash
mkdir <path_to_source_dir>/build
cd <path_to_source_dir>/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Python Bindings ###

The python bindings that are needed for integration with bingo can be made by
running the following commend from the build directory:

```bash
make bingocpp
```

A common error in the build of the python bindings is that the build must be
use the same version of python that will run your bingo scripts.  Pybind
usually finds the default python on your machine during build, so the easiest
way to ensure consistent python versioning is to build bingocpp in a Python 3
virtual environment.

### Documentation ###

Sphynx is used for automatically generating API documentation for bingo. The
most recent build of the documentation can be found in the repository at:
doc/_build/html/index.htm

## Running Tests ##

Several unit and integration tests can be performed upon building, to ensure a
proper install.  The test suite can be started by running the following command
from the build directory:

```bash
make gtest
```

## Usage Example ##

TODO

## Contributing ##

1. Fork it (<https://github.com/nasa/bingo/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Versioning ##

We use [SemVer](http://semver.org/) for versioning. For the versions available,
see the [tags on this repository](https://github.com/nasa/bingocpp/tags).

## Authors ##

* Geoffrey Bomarito
* Ethan Adams
* Tyler Townsend
  
## License ##

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
