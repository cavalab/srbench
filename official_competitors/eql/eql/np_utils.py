# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Original numpy utility functions."""

from jax import tree_util
import numpy as np
from scipy import special


# TODO(shoyer): Remove flatten() and unflatten() after they are checked in to
# jax.
def flatten(params, dtype=np.float64):
    """Flattens the params to 1d original numpy array.

    Args:
      params: pytree.
      dtype: the data type of the output array.

    Returns:
      (tree, shapes), vec
        * tree: the structure of tree.
        * shapes: List of tuples, the shapes of leaves.
        * vec: 1d numpy array, the flatten vector of params.
    """
    leaves, tree = tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    vec = np.concatenate([leaf.ravel() for leaf in leaves]).astype(dtype)
    return (tree, shapes), vec


def unflatten(spec, vec):
    """Unflattens the 1d original numpy array to pytree.

    Args:
      spec: (tree, shapes).
        * tree: the structure of tree.
        * shapes: List of tuples, the shapes of leaves.
      vec: 1d numpy array, the flatten vector of params.

    Returns:
      A pytree.
    """
    tree, shapes = spec
    sizes = [int(np.prod(shape)) for shape in shapes]
    leaves_flat = np.split(vec, np.cumsum(sizes)[:-1])
    leaves = [leaf.reshape(shape) for leaf, shape in zip(leaves_flat, shapes)]
    return tree_util.tree_unflatten(tree, leaves)
