"""
This module contains the abstract definition of the data containers that store
training data for bingo evolutionary analysis.
"""

import abc


class TrainingData(metaclass=abc.ABCMeta):
    """An index-able data containing class

    An abstract base class for a training data container.
    """
    @abc.abstractmethod
    def __getitem__(self, items):
        """This function allows for the sub-indexing of the training data

        Parameters
        ----------
        items : list
            indices for the subset

        Returns
        -------
        TrainingData :
            A subset of the original data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """Gets the number of indexable points in the training data

        Returns
        -------
        int :
            size of the training dataset
        """
        raise NotImplementedError
