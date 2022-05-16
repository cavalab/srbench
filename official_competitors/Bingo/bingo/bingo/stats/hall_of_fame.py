"""
The Hall of Fame captures the best individuals that have occurred in a
population's history.  The individuals in the hall of fame are kept in order of
ascending fitness/key value. If multiple individuals have the same fitness, the
earliest occurring is ordered first.  Thus, the earliest occurrence of the best
fitness/key value was achieved with the individual at index 0.

The hall of fame can -- for the most part -- be treated like a list: e.g.,
slicing, iterating and retrieving its length.
"""

from bisect import bisect_right
from copy import deepcopy
import numpy as np


class HallOfFame:
    """Keeping track of the best individuals in a population.

    The best individuals that have occurred in a population are tracked in the
    hall of fame.  They are kept in order, with the best/earliest occurring
    individuals first. 'Best' is quantified by the fitness value of the
    individual unless specified otherwise by the `key` parameter.  When a
    similarity function is given, the earliest occurrence of similar
    individuals is tracked.

    Parameters
    ----------
    max_size : int
        The maximum number of individuals to track.
    key_function : function (optional)
        Function used to quantify "best" individuals. Default is the use of an
        individual's fitness attribute. The signature of the function should be
        `func(chromosome)`
    similarity_function : function (optional)
        The function used to identify similar individuals. The signature of the
        function should be`func(chromosome, chromosome)`
    """

    def __init__(self, max_size, key_function=None, similarity_function=None):
        self._max_size = max_size
        self._similarity_func = similarity_function
        self._key_func = (lambda x: x.fitness) if key_function is None \
            else key_function
        self._keys = []
        self._items = []

    def insert(self, item):
        """Manually Insert an individual into the Hall of Fame.

        Inserts an individual, keeping correct order, into the hall of fame.
        The hall of fame will not be resized to maintain maximum size.

        Parameters
        ----------
        item : `Chromosome`
            The individual to be added to the hall of fame
        """
        item = deepcopy(item)
        item_key = self._key_func(item)
        index = bisect_right(self._keys, item_key)
        self._keys.insert(index, item_key)
        self._items.insert(index, item)

    def update(self, population):
        """Update the hall of fame based on the given population

        Parameters
        ----------
        population : list of `Chromosome`
            The list of individuals to be considered for induction into the
            hall of fame
        """
        for item in population:
            if self._item_should_be_added(item):
                if len(self) >= self._max_size:
                    self.remove(-1)
                self.insert(item)

    def _item_should_be_added(self, item):
        item_key = self._key_func(item)
        if np.isnan(item_key):
            return False
        if not self:
            return True
        if item_key <= self._keys[-1] or len(self) < self._max_size:
            return self._not_similar(item)
        return False

    def _not_similar(self, item):
        if self._similarity_func is None:
            return True
        for i in self._items:
            if self._similarity_func(i, item):
                return False
        return True

    def remove(self, index):
        """Remove a specific hall of fame member

        Parameters
        ----------
        index : int
            index in the hall of fame to be removed
        """
        del self._keys[index]
        del self._items[index]

    def clear(self):
        """Remove all hall of fame members"""
        del self._keys[:]
        del self._items[:]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __str__(self):
        return '\n'.join(["{}\t{}".format(key, i)
                          for key, i in zip(self._keys, self._items)])
