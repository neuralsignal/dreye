"""
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Collection
from inspect import getmembers, isfunction, signature

from dreye.err import DreyeError


def inherit_docstrings(cls):
    """
    Decorator to inherit docstrings of parent class.
    """
    for name, func in getmembers(cls, isfunction):
        if func.__doc__:
            continue
        for parent in cls.__mro__[1:]:
            if hasattr(parent, name):
                func.__doc__ = getattr(parent, name).__doc__
    return cls


class _AbstractArray(ABC):
    """
    Abstract Sequence
    """

    @abstractmethod
    def to_dict(self):
        """
        Convert object to dictionary.
        """
        pass

    @abstractmethod
    def from_dict(self):
        """
        Create instance from a dictionary.
        """
        pass


class _InitDict:
    """
    Implements `to_dict` and `from_dict` methods
    by using keyword arguments from the `__init__` method
    """

    def to_dict(self):
        """
        Return dictionary of `__init__` arguments.
        """
        keys = list(signature(self.__init__).parameters)
        return {key: getattr(self, key) for key in keys}

    @classmethod
    def from_dict(cls, data):
        """
        From dictionary build class
        """
        return cls(**data)


@inherit_docstrings
class CallableList(list):
    """
    Callable list type.

    This class works the same way as a native python list except that it has a
    `__call__` method that is applied to each element of the list.
    """

    def __init__(self, iterable, container_class=None):
        super().__init__(iterable)
        self._container_class = container_class

    def __call__(self, *args, iter_kwargs=None, **kwargs):
        """
        Calls each element in the list, and attempts to return
        a container-type class if they match the allowed objects.

        Parameters
        ----------
        args : tuple
            Positional arguments passed directly to call
            of each element.
        iter_kwargs : list of dicts
            List of keyword argument for each element individually
        kwargs : dict
            Keyword argument passed directly to call of each element.

        Returns
        -------
        container : list-like or container-type class
            The list of the objects after having been called
        """
        container = [
            ele(*args, **kwargs)
            if iter_kwargs is None
            else ele(*args, **kwargs, **iter_kwargs[index])
            for index, ele in enumerate(self)
        ]
        if self._container_class is None:
            return CallableList(
                container,
                self._container_class
            )
        elif self._container_class._are_instances(container):
            return self._container_class(container)
        else:
            return CallableList(
                container,
                self._container_class
            )


class _AbstractContainer(ABC):
    """
    The abstract container allows storage of specific objects and
    access to their attributes and methods.
    """

    _init_keys = []
    _enforce_instance = None

    def __init__(self, container=[]):
        self._container = self._check_list(container)
        self._init_attrs()

    def __copy__(self):
        return type(self)([ele.copy() for ele in self])

    def copy(self):
        """
        Copy container.

        Returns
        -------
        container : object
            A copy of the container.
        """
        return copy.copy(self)

    def _init_attrs(self):
        for key in self._init_keys:
            setattr(self, key, None)

    @property
    @abstractmethod
    def _allowed_instances(self):
        pass

    @classmethod
    def _are_instances(cls, container):
        return all([
            isinstance(ele, cls._allowed_instances)
            for ele in container
        ])

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"{'; '.join([ele.__repr__() for ele in self])})"
        )

    def __str__(self):
        joined_elements = ';\n\t'.join([ele.__str__() for ele in self])
        return (
            f"{type(self).__name__}(\n\t"
            f"{joined_elements})"
        )

    @property
    def container(self):
        """
        Python `list` of container.
        """
        return self._container

    @container.setter
    def container(self, container):
        self.__init__(container)

    def __iter__(self):
        return iter(self.container)

    def __len__(self):
        return len(self.container)

    def len(self):
        """
        Length of container.

        Returns
        -------
        length : int
            Number of elements in container.
        """
        return len(self.container)

    def __getitem__(self, key):
        return self.container[key]

    def __setitem__(self, key, value):
        value = self._check_ele(value)
        self._container[key] = value
        self._init_attrs()

    def __call__(self, *args, iter_kwargs=None, **kwargs):
        """
        Use the `__call__` method on each element in container

        Parameters
        ----------
        args : tuple
            Positional arguments passed directly to call
            of each element.
        iter_kwargs : list of dicts
            List of keyword argument for each element individually
        kwargs : dict
            Keyword argument passed directly to call of each element.

        Returns
        -------
        container : list-like or container-type class
            The list of the objects after having been called
        """
        container = [
            ele(*args, **kwargs)
            if iter_kwargs is None
            else ele(*args, **kwargs, **iter_kwargs[index])
            for index, ele in enumerate(self)
        ]
        if self._are_instances(container):
            return type(self)(container)
        else:
            return CallableList(container, container_class=type(self))

    def __getattr__(self, name):
        if '_container' in vars(self):
            container = [getattr(ele, name) for ele in self]
            if self._are_instances(container):
                return type(self)(container)
            else:
                return CallableList(container, container_class=type(self))
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has "
                f"not attribute '{name}'."
            )

    def to_dict(self):
        """
        Returns Python list of container.
        """
        return self.container

    @classmethod
    def from_dict(cls, data):
        """
        Creates container instance from python list.
        """
        return cls(data)

    def _check_list(self, container):
        if isinstance(container, Collection):
            container = list(container)
        else:
            raise DreyeError(
                'container must be a list-like.')
        if not self._are_instances(container):
            raise DreyeError(
                'not all elements of container are '
                f'of a {self._allowed_instances} instance.'
            )

        return container

    def _check_ele(self, ele):
        if not isinstance(ele, self._allowed_instances):
            raise DreyeError(
                f'New element of type {type(ele)} is'
                f'not a {self._allowed_instances} instance.'
            )
        return ele

    def append(self, value):
        """
        Append allowed object to container inplace

        Parameters
        ----------
        value : type
            An allowed object.
        """
        value = self._check_ele(value)
        self._container.append(value)
        self._init_attrs()

    def extend(self, value):
        """
        Extend a list of allowed objects to container inplace.

        Parameters
        ----------
        value : list-like
            A list of allowed objects.
        """
        value = self._check_list(value)
        self._container.extend(value)
        self._init_attrs()

    def pop(self, index=-1):
        """
        Remove an element/object from the container.

        Parameters
        ----------
        index : int
            The index of the element to remove.
        """
        value = self._container.pop(index)
        self._init_attrs()
        return value
