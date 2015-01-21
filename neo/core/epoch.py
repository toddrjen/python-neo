# -*- coding: utf-8 -*-
"""
This module defines :class:`Epoch`, an array of epochs.

:class:`Epoch` derives from :class:`Event`, from
:module:`neo.core.event`.
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys

import numpy as np

from neo.core.event import Event

PY_VER = sys.version_info[0]


def _new_epoch(cls, times=None, labels=None, durations=None, units=None,
               dtype=None, copy=None, name=None, file_origin=None,
               description=None, annotations=None):
    """Map :meth:`Epoch.__new__` to a function that does not do unit checking.

    This is needed for pickle to work.
    """
    if annotations is None:
        annotations = {}
    return cls(times=times, labels=labels, durations=durations, units=units,
               dtype=dtype, copy=copy, name=name, file_origin=file_origin,
               description=description, **annotations)


class Epoch(Event):

    """One or more epochs.

    Each epoch has with a time it occurs, an (optional) description,
    and an (optional) duration.

    *Usage*::

        >>> from neo.core import Epoch
        >>> from quantities import s, ms
        >>> import numpy as np
        >>>
        >>> epc = Epoch(times=np.arange(0, 30, 10)*s,
        ...             durations=[10, 5, 7]*ms,
        ...             labels=np.array(['btn0', 'btn1', 'btn2'], dtype='S'))
        >>>
        >>> epc.times
        array([  0.,  10.,  20.]) * s
        >>> epc.durations
        array([ 10.,   5.,   7.]) * ms
        >>> epc.labels
        array(['btn0', 'btn1', 'btn2'],
              dtype='|S4')

    *Required attributes/properties*:
        :times: (quantity array 1D) The starts of the time periods.
        :durations: (quantity array 1D) The length of the time period.
        :labels: (numpy.array 1D dtype='S') Names or labels for the
            time periods.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset,
        :description: (str) Text description,
        :file_origin: (str) Filesystem path or URL of the original data file.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`,

    """

    _necessary_attrs = (Event._necessary_attrs +
                        (('labels', np.ndarray, 1, np.dtype('S')),))
    _quantity_slice_attrs = ('durations',)

    def __new__(cls, times=None, labels=None, durations=None, units=None,
                dtype=None, copy=True, name=None, file_origin=None,
                description=None, **annotations):
        """Construct new :class:`Epoch` from data.

        This is called whenever a new class:`Epoch` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        """
        obj = Event.__new__(cls, times=times, labels=labels, units=units,
                            dtype=dtype, copy=copy, name=name,
                            file_origin=file_origin,
                            description=description)

        if durations is None:
            durations = np.zeros_like(obj.magnitude) * obj.units
        elif not hasattr(durations, 'units'):
            durations = durations * obj.units
        else:
            durations = durations.rescale(obj.units)
        obj.durations = durations

        return obj

    def __init__(self, times=None, labels=None, durations=None, units=None,
                 dtype=None, copy=True, name=None, file_origin=None,
                 description=None, **annotations):
        """Initialize a newly constructed :class:`Epoch` instance.

        This method is only called when constructing a new Epoch,
        not when slicing or viewing. We use the same call signature
        as __new__ for documentation purposes. Anything not in the call
        signature is stored in annotations.
        """
        Event.__init__(self, times=times, labels=labels, units=units,
                       dtype=dtype, copy=copy, name=name,
                       file_origin=file_origin, description=description,
                       **annotations)

    @property
    def _new_wrapper_args(self):
        """Return a tuple of the arguments used to construct _new_wrapper."""
        #      (times, labels, durations, units,
        #       dtype, copy, name, file_origin,
        #       description, **anotations)
        return (np.array(self), self.labels, self.durations, self.units,
                self.dtype, True, self.name, self.file_origin,
                self.description, self.annotations)

    def __repr__(self):
        """Return a string representing the :class:`Epoch`."""
        # use _safe_labels or repr is messed up
        objs = ['%s@%s for %s' % (label, time, dur) for label, time, dur in
                zip(self._safe_labels, self.times, self.durations)]
        return '<Epoch: %s>' % ', '.join(objs)
