# -*- coding: utf-8 -*-
"""This module implements the generic array base class.

All neo array objects inherit from it.  It provides shared methods for all
array types.

:class:`NeoArray` derives from :class:`BaseNeo`, from
:module:`neo.core.baseneo`, and from :class:`quantites.Quantity`, which
inherits from :class:`numpy.array`.

Inheritance from :class:`numpy.array` is explained here:
http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq

from neo.core.baseneo import BaseNeo, merge_annotations


def _new_neoarray(cls, data, units=None, dtype=None, copy=True,
                  name=None, file_origin=None, description=None,
                  arg1=None, arg2=None, arg3=None,
                  annotations=None):
    """Map NeoArray.__new__ to function that does not do the unit checking.

    This is needed for pickle to work.

    This is a stub version for documentation purposes.  Each class that derives
    from :class:NeoArray must implement its own and put it in the
    :attr:`_new_wrapper` attribute in the class.

    Replace `arg1=None, arg2=None, arg3=None,` with the additional arguments
    of the class.
    """
    # Remove the following exception when reimplementing this.
    raise NotImplementedError('This must be implemented individually for each '
                              'subclass of NeoArray')

    # the lines below this one must be used when reimplementing this.
    # Replace `arg1=arg1, arg2=arg2, arg3=arg3,` with the additional arguments
    # of the class.
    if annotations is None:
        annotations = {}
    return cls(data=data, units=units, dtype=dtype, copy=copy,
               name=name, file_origin=file_origin, description=description,
               arg1=arg1, arg2=arg2, arg3=arg3,
               **annotations)


class NeoArray(BaseNeo, pq.Quantity):

    """The base class from which Neo Array objects inherit.

    It derives from :class:`BaseNeo` and :class:`quantities.Quantity`.

    In addition to the setup :class:`BaseNeo` does, this class also
    automatically sets up arrays to hold additional data.

    Each class can define one or more of the following class attributes
    (in  addition to those of :class:`BaseNeo` and
    :class:`quantities.Quantity`):
        :_main_attr: The name of the attribute that should return the object
                     itself.  This is created automatically as a read-only
                     property, unless it is 'times', 't_start', 't_stop', or
                     'duration', all of which have to be implemented manually
                     (see below)
        :_quantity_slice_attrs: The names of attributes containing
                                :class:`quantities.Quantity` arrays that should
                                be sliced when the object is sliced.
        :_ndarray_slice_attrs: The names of attributes containing
                               :class:`numpy.ndarray` arrays that should
                                be sliced when the object is sliced.
        :_quantity_match_attrs: The name of attributes containing
                                :class:`quantities.Quantity` objects that
                                should have their dimensionality matched
                                to that of :attr:`_main_attr`.  At present
                                this only happens on initialization.
        :_consistency_attrs: These attributes must be the same between two
                             objects for them to be considered consistent
                             and thus combinable.
        :_private_attrs: These attributes are private and not visible to
                         users, but should still be preserved when doing
                         operations like copies.
        :_default_vals: A dictionary where the key is any attribute and the
                        value is the default value for that attribute.
                        An attribute only needs to be specified here if its
                        default value is not `None`.
        :_required_dimensionality: If a specific dimensionality is required,
                                   specify it here.
        :_new_wrapper: The reimplementation of the `_new_neoarray` function for
                       the class. It is needed for pickling.

    The following helper properties MUST be reimplemented by any class that
    subclasses :class:`NeoArray`:
        :t_start: The relative start time of the data.  This must be
                  reimplemented in any class that derives from
                  :class:`BaseNeo`.
        :t_stop: The relative stop time of the data.  This must be
                 reimplemented in any class that derives from
                 :class:`BaseNeo`.
        :times: The relative time points of the current array.  This must be
                reimplemented in any class that derives from :class:`BaseNeo`.
        :duration: The duration of the data.  This must be reimplemented in
                   any class that derives from :class:`BaseNeo`.
        :_new_wrapper_args: Return arguments for :attr:`_new_wrapper`.
                            This should return a tuple of arguments in the same
                            order as the arguments for :attr:`_new_wrapper`.
                            It must NOT return a dictionary of arguments.

    The following helper properties are available
    (in  addition to those of :class:`BaseNeo` and
    :class:`quantities.Quantity`):
        :_slice_attrs: :attrs:`_ndarray_slice_attrs` +
                       :attrs:`_quantity_slice_attrs`
    Additionally, a read-only property with the name specified in :_main_attr:
    is created that returns the object itself.

    The following "universal" methods are available
    (in  addition to those of :class:`BaseNeo` and
    :class:`quantities.Quantity`):
        :duplicate_with_new_data(newdata): Create a copy with the properties
                                            of the current object but the data
                                            from `newdata`.
        :sort: Sort the data and any :attr:`_slice_attrs` by the data values.


    Each child class should:
        0) call NeoArray.__init__(self, name=name, description=description,
                                  file_origin=file_origin, **annotations)
           with the universal recommended arguments, plus optional annotations
        1) call NeoArray.__new__(self, name=name, description=description,
                                  file_origin=file_origin, **annotations)
           with the universal recommended arguments, plus optional annotations
        1) process its required arguments in its __new__ method
        2) process its non-universal recommended arguments in its __new__
           method.
        3) implemenet t_start, t_stop, times, and duration.
        4) implement _new_wrapper_args
    """

    _main_attr = None
    _quantity_slice_attrs = ()
    _ndarray_slice_attrs = ()
    _quantity_match_attrs = ()
    _consistency_attrs = ()
    _private_attrs = ()
    _default_vals = {}
    _required_dimensionality = None
    _new_wrapper = _new_neoarray

    def __new__(cls, data, units=None, dtype=None, copy=True,
                name=None, file_origin=None, description=None,
                **annotations):
        """Construct new :class:`NeoArray` from data.

        This is called whenever a new class:`NeoArray` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        """
        if copy is None:
            copy = True

        # Make sure units are consistent
        # also get the dimensionality now since it is much faster to feed
        # that to Quantity rather than a unit
        if units is None:
            # No keyword units, so get from `data`
            try:
                units = data.units
                dim = units.dimensionality
            except AttributeError:
                raise ValueError('you must specify units')
        else:
            if hasattr(units, 'dimensionality'):
                dim = units.dimensionality
            else:
                dim = pq.quantity.validate_dimensionality(units)

            if (hasattr(data, 'dimensionality') and
                    data.dimensionality.items() != dim.items()):
                if not copy:
                    raise ValueError("cannot rescale and return view")
                else:
                    # this is needed because of a bug in python-quantities
                    # see issue # 65 in python-quantities github
                    # remove this if it is fixed
                    data = data.rescale(dim)

        if dtype is None:
            dtype = getattr(data, 'dtype', np.float)
        elif hasattr(data, 'dtype') and data.dtype != dtype:
            if not copy:
                raise ValueError("cannot change dtype and return view")

        # check to make sure the units are correct
        # this approach is orders of magnitude faster than comparing the
        # reference dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            ValueError("Unit %s has dimensions %s, not [%s]" %
                       (units, dim.simplified))

        obj = pq.Quantity.__new__(cls, data, units=dim, dtype=dtype,
                                  copy=copy)

        obj.name = name
        obj.file_origin = file_origin
        obj.description = description

        for attr in cls._single_parent_objects:
            setattr(obj, attr, None)
        for attr in cls._multi_parent_objects:
            setattr(obj, attr, [])

        return obj

    def __init__(self, data, units=None, dtype=None, copy=True,
                 name=None, file_origin=None, description=None,
                 **annotations):
        """Initialize a newly constructed :class:`NeoArray` instance.

        This method is only called when constructing a new NeoArray,
        not when slicing or viewing. We use the same call signature
        as __new__ for documentation purposes. Anything not in the call
        signature is stored in annotations.
        """
        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        BaseNeo.__init__(self, units=units, dtype=dtype, copy=copy,
                         name=name, file_origin=file_origin,
                         description=description, **annotations)

        # Create an attribute that return the object itself.
        # Don't create it if it is one of the
        if self._main_attr not in ('times', 't_start', 't_stop', 'duration'):
            setattr(self, self._main_attr, self._return_self)

    @property
    def _slice_attrs(self):
        """The names of all array-type attributes."""
        return self._ndarray_slice_attrs + self._quantity_slice_attrs

    @property
    def _new_wrapper_args(self):
        """Return a tuple of the arguments used to construct _new_wrapper."""
        raise NotImplementedError('t_start must be implemented in a subclass')
        #      (data, units, dtype,copy, name,
        #       file_origin, description, **anotations)
        return (np.array(self), self.units, self.dtype, True, self.name,
                self.file_origin, self.description, self.annotations)

    def __reduce__(self):
        """Map the __new__ function onto _new_neoarray so that pickle works."""
        return self._new_wrapper, self._new_wrapper_args

    def __array_finalize__(self, obj):
        """Put the array data in the final form it needs for use.

        This is called every time a new :class:`NeoArray` is created.

        It is the appropriate place to set default values for attributes
        for :class:`NeoArray` constructed by slicing or viewing.

        User-specified values are only relevant for construction from
        constructor, and these are set in __new__. Then they are just
        copied over here.
        """
        super(NeoArray, self).__array_finalize__(obj)

        # Supposedly, during initialization from constructor, obj is supposed
        # to be None, but this never happens. It must be something to do
        # with inheritance from Quantity.
        if obj is None:
            return

        self._copy_data_complement(obj)

    def _check_consistency(self, other):
        """Check if the attributes of another NeoArray are compatible."""
        if isinstance(other, self.__class__):
            for attr in self._consistency_attrs:
                if getattr(self, attr) != getattr(other, attr):
                    raise ValueError("Inconsistent values of %s" % attr)
            # how to handle name and annotations?

    def _copy_data_complement(self, other):
        """Copy the metadata from another :class:`NeoArray`."""
        for attr in (self._necessary_attrs +
                     self._recommended_attrs +
                     self._private_attrs +
                     ('lazy_shape',)):
            setattr(self, attr,
                    getattr(other, attr, self._default_vals.get(attr, None)))

        if hasattr(other, 'lazy_shape'):
            self.lazy_shape = other.lazy_shape

    def _apply_operator(self, other, op, *args):
        """Handle mathematical operations.

        Makes sure metadata is copied to the new :class:`NeoArray`.
        """
        self._check_consistency(other)
        f = getattr(super(NeoArray, self), op)
        new_neoarray = f(other, *args)
        new_neoarray._copy_data_complement(self)
        return new_neoarray

    @property
    def _return_self(self):
        """Return the object itself."""
        return self

    @property
    def t_start(self):
        """Time when data begins."""
        raise NotImplementedError('t_start must be implemented in a subclass')

    @property
    def duration(self):
        """Signal duration."""
        raise NotImplementedError('t_start must be implemented in a subclass')

    @property
    def t_stop(self):
        """Time when data ends."""
        raise NotImplementedError('t_start must be implemented in a subclass')

    @property
    def times(self):
        """The time points of each sample of the data."""
        raise NotImplementedError('t_start must be implemented in a subclass')

    def rescale(self, units):
        """Return a copy of the NeoArray converted to the specified units."""
        to_dims = pq.quantity.validate_dimensionality(units)
        if self.dimensionality == to_dims:
            to_u = self.units
            data = np.array(self)
        else:
            to_u = pq.Quantity(1.0, to_dims)
            from_u = pq.Quantity(1.0, self.dimensionality)
            try:
                cf = pq.quantity.get_conversion_factor(from_u, to_u)
            except AssertionError:
                raise ValueError('Unable to convert between units of "%s" \
                                 and "%s"' % (from_u._dimensionality,
                                              to_u._dimensionality))
            data = cf * self.magnitude
        new = self.__class__(data=data, units=to_u,
                             sampling_rate=self.sampling_rate)
        new._copy_data_complement(self)
        new.annotations.update(self.annotations)
        return new

    def sort(self):
        """Sort by the main data.

        Both the :class:`NeoArray` and its :attr:`_slice_attrs`, if any,
        are sorted.
        """
        # sort the waveforms by the times
        sort_indices = np.argsort(self)
        for attr in self._slice_attrs:
            value = getattr(self, attr)
            if value is None or not value.any():
                continue
            setattr(self, attr, value[sort_indices])

        # now sort the times
        # We have sorted twice, but `self = self[sort_indices]` introduces
        # a dependency on the slicing functionality of SpikeTrain.
        super(NeoArray, self).sort()

    def duplicate_with_new_data(self, newdata):
        """Create a new NeoArray with the same metadata but different data."""
        # newdata is the new data
        new = self.__class__(data=newdata, units=self.units)
        new._copy_data_complement(self)
        new.annotations.update(self.annotations)
        return new

    def merge(self, other):
        """Merge the another :class:`NeoArray` into this one.

        A new object of the same class is returned.
        The :class:`NeoArray` objects are concatenated horizontally
        (column-wise), :func:`np.hstack`).

        If the attributes of the two :class:`NeoArray` are not
        compatible, and Exception is raised.
        """
        self._check_consistency(other)
        other = other.rescale(self.units)
        new = self.duplicate_with_new_data(np.hstack([self, other])*self.units)
        for attr in self._quantity_slice_attrs:
            targunits = getattr(self, attr).units
            newarr = np.hstack([getattr(self, attr),
                                getattr(other, attr)]) * targunits
            setattr(new, attr, newarr)

        for attr in self._ndarray_slice_attrs:
            newarr = np.hstack([getattr(self, attr), getattr(other, attr)])
            setattr(new, attr, newarr)

        new.annotations = merge_annotations(self.annotations,
                                            other.annotations)

        return new

    def __eq__(self, other):
        """Equality test (==)."""
        if not self._check_consistency(other):
            return False
        return super(NeoArray, self).__eq__(other)

    def __ne__(self, other):
        """Non-equality test (!=)."""
        return not self.__eq__(other)

    def __add__(self, other, *args):
        """Addition (+)."""
        return self._apply_operator(other, "__add__", *args)

    def __sub__(self, other, *args):
        """Subtraction (-)."""
        return self._apply_operator(other, "__sub__", *args)

    def __mul__(self, other, *args):
        """Multiplication (*)."""
        return self._apply_operator(other, "__mul__", *args)

    def __truediv__(self, other, *args):
        """Float division (/)."""
        return self._apply_operator(other, "__truediv__", *args)

    def __div__(self, other, *args):
        """Integer division (//)."""
        return self._apply_operator(other, "__div__", *args)

    __radd__ = __add__
    __rmul__ = __sub__

    def __rsub__(self, other, *args):
        """Backward subtraction (other-self)."""
        return self.__mul__(-1, *args) + other

    def __getslice__(self, *args, **kargs):
        """Get a slice.

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        """
        obj = super(NeoArray, self).__getslice__(*args, **kargs)
        # somehow this knows to call NeoArray.__array_finalize__, though
        # I'm not sure how. (If you know, please add an explanatory comment
        # here.) That copies over all of the metadata.

        for attr in self._slice_attrs:
            value = getattr(self, attr, None)
            if value is None or not value.size:
                continue
            setattr(obj, attr, value.__getslice__(*args, **kargs))
        return obj

    def __getitem__(self, *args, **kargs):
        """Get an item or slice."""
        obj = super(NeoArray, self).__getitem__(*args, **kargs)
        for attr in self._slice_attrs:
            value = getattr(self, attr, None)
            if value is None or not value.size:
                continue
            setattr(obj, attr, value.__getitem__(*args, **kargs))
        return obj

    def _repr_pretty_(self, pp, cycle):
        """Handle pretty-printing."""
        pp.text(" ".join([self.__class__.__name__,
                          "in",
                          str(self.units),
                          "with",
                          "x".join(map(str, self.shape)),
                          str(self.dtype),
                          "values",
                          ]))

        if self._has_repr_pretty_attrs_():
            pp.breakable()
            self._repr_pretty_attrs_(pp, cycle)


def _get_sampling_rate(sampling_rate, sampling_period):
    """Get the sampling_rate.

    This is calculated from either the sampling_period or the
    sampling_rate.

    If both are specified, make sure they match.  If not a ValueError is
    raised.
    """
    if sampling_period is None:
        if sampling_rate is None:
            raise ValueError("You must provide either the sampling rate or " +
                             "sampling period")
    elif sampling_rate is None:
        sampling_rate = 1.0 / sampling_period
    elif sampling_period != 1.0 / sampling_rate:
        raise ValueError('The sampling_rate has to be 1/sampling_period')
    if not hasattr(sampling_rate, 'units'):
        raise TypeError("Sampling rate/sampling period must have units")
    return sampling_rate


def _check_has_dimensions_time(*values):
    """Verify that values have a dimensionality compatible with time."""
    errmsgs = []
    for value in values:
        dim = value.dimensionality
        if (len(dim) != 1 or list(dim.values())[0] != 1 or
                not isinstance(list(dim.keys())[0], pq.UnitTime)):
            errmsgs.append("value %s has dimensions %s, not [time]" %
                           (value, dim.simplified))
    if errmsgs:
        raise ValueError("\n".join(errmsgs))


def _check_time_in_range(value, t_start, t_stop, view=False):
    """Verify that all times in a quantity array are between two values.

    It check that values in :attr:`value` are between :attr:`t_start`
    and :attr:`t_stop` (inclusive) and raises a ValueError if they are not.

    If :attr:`view` is True, vies are used for the test.
    Using drastically increases the speed, but is only safe if you are
    certain that the dtype and units are the same
    """
    if not value.size:
        return

    if view:
        value = value.view(np.ndarray)
        t_start = t_start.view(np.ndarray)
        t_stop = t_stop.view(np.ndarray)

    if value.min() < t_start:
        raise ValueError("The first spike (%s) is before t_start (%s)" %
                         (value, t_start))
    if value.max() > t_stop:
        raise ValueError("The last spike (%s) is after t_stop (%s)" %
                         (value, t_stop))
