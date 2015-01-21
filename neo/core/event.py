# -*- coding: utf-8 -*-
"""This module defines :class:`Event`.

:class:`Event` derives from :class:`NeoArray`, from
:module:`neo.core.neoarray`.
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import sys

import numpy as np
import quantities as pq

from neo.core.neoarray import NeoArray

PY_VER = sys.version_info[0]


def _new_event(cls, times, labels=None, units=None, dtype=None, copy=None,
               name=None, file_origin=None, description=None,
               annotations=None):
    """Map :meth:`Event.__new__` to a function that does not do unit checking.

    This is needed for pickle to work.
    """
    if annotations is None:
        annotations = {}
    return cls(times=times, labels=labels, units=units, dtype=dtype, copy=copy,
               name=name, file_origin=file_origin, description=description,
               **annotations)


class Event(NeoArray):

    """One or more events.

    Each event has with a time it occurs and an (optional) description.

    *Usage*::

        >>> from neo.core import Event
        >>> from quantities import s
        >>> import numpy as np
        >>>
        >>> evt = Event(np.arange(0, 30, 10)*s,
        ...             labels=np.array(['trig0', 'trig1', 'trig2'],
        ...                             dtype='S'))
        >>>
        >>> evt.times
        array([  0.,  10.,  20.]) * s
        >>> evt.labels
        array(['trig0', 'trig1', 'trig2'],
              dtype='|S5')

    *Required attributes/properties*:
        :times: (quantity array 1D) The time of the events.
        :labels: (numpy.array 1D dtype='S') Names or labels for the events.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    """

    _main_attr = 'times'
    _single_parent_objects = ('Segment', 'Unit', 'RecordingChannel', 'Block')
    _necessary_attrs = (('labels', np.ndarray, 1, np.dtype('S')),)
    _ndarray_slice_attrs = ('labels',)
    _required_dimensionality = pq.UnitTime
    _new_wrapper = _new_event

    def __new__(cls, times, labels=None, units=None, dtype=None,
                copy=True, name=None, file_origin=None, description=None,
                **annotations):
        """Construct new :class:`Event` from data.

        This is called whenever a new class:`Event` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        """
        obj = NeoArray.__new__(cls, data=times, units=units,
                               dtype=dtype, copy=copy, name=name,
                               file_origin=file_origin,
                               description=description)

        if labels is None:
            labels = np.array(['']*len(times), dtype='S')
        obj.labels = labels

        return obj

    def __init__(self, times, labels=None, units=None, dtype=None,
                 copy=True, name=None, file_origin=None, description=None,
                 **annotations):
        """Initialize a newly constructed :class:`Event` instance.

        This method is only called when constructing a new Event,
        not when slicing or viewing. We use the same call signature
        as __new__ for documentation purposes. Anything not in the call
        signature is stored in annotations.
        """
        NeoArray.__init__(self, data=times, units=units, dtype=dtype,
                          copy=copy, name=name, file_origin=file_origin,
                          description=description, **annotations)

    @property
    def _new_wrapper_args(self):
        """Return a tuple of the arguments used to construct _new_wrapper."""
        #      (times, labels, units, dtype, copy,
        #       name, file_origin, description,
        #       **anotations)
        return (np.array(self), self.labels, self.units, self.dtype, True,
                self.name, self.file_origin, self.description,
                self.annotations)

    def __repr__(self):
        """Return a string representing the :class:`Event`."""
        # use _safe_labels or repr is messed up
        objs = ['%s@%s' % (label, time) for label, time in
                zip(self._safe_labels, self.times)]
        return '<Event: %s>' % ', '.join(objs)

    @property
    def _safe_labels(self):
        """Convert labels to unicode for python 3."""
        if PY_VER == 3:
            return self.labels.astype('U')
        else:
            return self.labels

    @property
    def t_start(self):
        """Time when data begins."""
        return np.min(self)

    @property
    def duration(self):
        """Signal duration."""
        return self.t_stop - self.t_start

    @property
    def t_stop(self):
        """Time when data ends."""
        return np.min(self)

    @property
    def times(self):
        """The time points of each sample of the data."""
        return self
