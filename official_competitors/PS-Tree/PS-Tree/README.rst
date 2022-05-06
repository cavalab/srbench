==================
PS-Tree
==================


.. image:: https://img.shields.io/pypi/v/pstree.svg
        :target: https://pypi.python.org/pypi/pstree

.. image:: https://img.shields.io/travis/hengzhe-zhang/pstree.svg
        :target: https://travis-ci.com/hengzhe-zhang/pstree

.. image:: https://readthedocs.org/projects/pstree/badge/?version=latest
        :target: https://pstree.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




An open source python library for non-linear piecewise symbolic regression based on Genetic Programming


* Free software: MIT license
* Documentation: https://pstree.readthedocs.io.

Introduction
----------------
Piece-wise non-linear regression is a long-standing problem in the machine learning domain that has long plagued machine learning researchers. It is extremely difficult for users to determine the correct partition scheme and non-linear model when there is no prior information. To address this issue, we proposed piece-wise non-linear regression tree (PS-Tree), an automated piece-wise non-linear regression method based on decision tree and genetic programming techniques. Based on such an algorithm framework, our method can produce an explainable model with high accuracy in a short period of time.

Installation
----------------

.. code:: bash

    pip install -U pstree

Features
----------------

* A fully automated piece-wise non-linear regression tool
* A fast genetic programming based symbolic regression tool

Example
----------------
An example of usage:

.. code:: Python

    X, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    r = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                        height_limit=6, n_pop=25, n_gen=100,
                        basic_primitive='optimal', size_objective=True)
    r.fit(x_train, y_train)
    print(r2_score(y_test, r.predict(x_test)))

Experimental results on SRBench:

.. image:: https://raw.githubusercontent.com/hengzhe-zhang/PS-Tree/master/docs/R2-result.png

Credits
--------------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
