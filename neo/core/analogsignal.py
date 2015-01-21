# -*- coding: utf-8 -*-
"""This module implements :class:`AnalogSignal`, an n-dimensional signal.

:class:`IrregularlySampledSignal` is not derived from :class:`AnalogSignal`
and is defined in :module:`neo.core.irregularlysampledsignal`.

:class:`AnalogSignal` inherits from :class:`NeoArray`.
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq

from neo.core.neoarray import NeoArray, _get_sampling_rate


def _new_analogsignal(cls, signal, units=None, dtype=None, copy=True,
                      t_start=0*pq.s, sampling_rate=None,
                      sampling_period=None, name=None, file_origin=None,
                      description=None, channel_index=None,
                      annotations=None):
    """Map AnalogSignal.__new__ to function that does not do the unit checking.

    This is needed for pickle to work.
    """
    return cls(signal=signal, units=units, dtype=dtype, copy=copy,
               t_start=t_start, sampling_rate=sampling_rate,
               sampling_period=sampling_period, name=name,
               file_origin=file_origin, description=description,
               channel_index=channel_index,
               **annotations)


class AnalogSignal(NeoArray):

    """A continuous, n-dimensional analog signal.

    A representation of a continuous, analog signal acquired at time
    :attr:`t_start` at a certain sampling rate.

    Inherits from :class:`NeoArray`.

    *Usage*::

        >>> from neo.core import AnalogSignal
        >>> from quantities import kHz, ms, nA, s, uV
        >>> import numpy as np
        >>>
        >>> sig0 = AnalogSignal([1, 2, 3], sampling_rate=0.42*kHz,
        ...                     units='mV')
        >>> sig1 = AnalogSignal([4, 5, 6]*nA, sampling_period=42*ms)
        >>> sig2 = AnalogSignal(np.array([1.0, 2.0, 3.0]), t_start=42*ms,
        ...                     sampling_rate=0.42*kHz, units=uV)
        >>> sig3 = AnalogSignal([1], units='V', day='Monday',
        ...                     sampling_period=1*s)
        >>>
        >>> sig3
        <AnalogSignal(array([1]) * V, [0.0 s, 1.0 s], sampling rate: 1.0 1/s)>
        >>> sig3.annotations['day']
        'Monday'
        >>> sig3[0]
        array(1) * V
        >>> sig3[::2]
        <AnalogSignal(array([1]) * V, [0.0 s, 2.0 s], sampling rate: 0.5 1/s)>

    *Required attributes/properties*:
        :signal: (quantity array 1D, numpy array 1D, or list) The data itself.
        :units: (quantity units) Required if the signal is a list or NumPy
                array, not if it is a :class:`Quantity`
        :sampling_rate: *or* :sampling_period: (quantity scalar) Number of
                                               samples per unit time or
                                               interval between two samples.
                                               If both are specified, they are
                                               checked for consistency.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :t_start: (quantity scalar) Time when signal begins.
            Default: 0.0 seconds
        :channel_index: (int) You can use this to order :class:`AnalogSignal`
            objects in an way you want.  :class:`AnalogSignalArray` and
            :class:`Unit` objects can be given indexes as well so related
            objects can be linked together.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
        :copy: (bool) True by default.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_rate: (quantity scalar) Number of samples per unit time.
            (1/:attr:`sampling_period`)
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`sampling_rate`)
        :duration: (quantity scalar) Signal duration, read-only.
            (:attr:`size` * :attr:`sampling_period`)
        :t_stop: (quantity scalar) Time when signal ends, read-only.
            (:attr:`t_start` + :attr:`duration`)
        :times: (quantity 1D) The time points of each sample of the signal,
            read-only.
            (:attr:`t_start` + arange(:attr:`shape`)/:attr:`sampling_rate`)

    *Slicing*:
        :class:`AnalogSignal` objects can be sliced. When this occurs, a new
        :class:`AnalogSignal` (actually a view) is returned, with the same
        metadata, except that :attr:`sampling_period` is changed if
        the step size is greater than 1, and :attr:`t_start` is changed if
        the start index is greater than 0.  Getting a single item
        returns a :class:`~quantity.Quantity` scalar.

    *Operations available on this object*:
        == != + * /

    """

    _single_parent_objects = ('Segment', 'RecordingChannel', 'Block')
    _main_attr = 'signal'
    _necessary_attrs = (('sampling_rate', pq.Quantity, 0),
                        ('t_start', pq.Quantity, 0))
    _recommended_attrs = ((('channel_index', int),) +
                          NeoArray._recommended_attrs)
    _default_vals = {'_t_start': 0 * pq.s}
    _consistency_attrs = ("t_start", "sampling_rate")

    def __new__(cls, signal, units=None, dtype=None, copy=True,
                t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                name=None, file_origin=None, description=None,
                channel_index=None, **annotations):
        """Construct new :class:`AnalogSignal` from data.

        This is called whenever a new class:`AnalogSignal` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        """
        if units is None:
            if hasattr(signal, "units"):
                units = signal.units
            else:
                raise ValueError("Units must be specified")
        elif isinstance(signal, pq.Quantity):
            # could improve this test, what if units is a string?
            if units != signal.units:
                signal = signal.rescale(units)
        obj = pq.Quantity.__new__(cls, signal, units=units, dtype=dtype,
                                  copy=copy)

        if t_start is None:
            raise ValueError('t_start cannot be None')
        obj._t_start = t_start

        obj._sampling_rate = _get_sampling_rate(sampling_rate, sampling_period)

        obj.channel_index = channel_index
        obj.segment = None
        obj.recordingchannel = None

        return obj

    def __init__(self, signal, units=None, dtype=None, copy=True,
                 t_start=0 * pq.s, sampling_rate=None, sampling_period=None,
                 name=None, file_origin=None, description=None,
                 channel_index=None, **annotations):
        """Initialize a newly constructed :class:`AnalogSignal` instance.

        This method is only called when constructing a new AnalogSignal,
        not when slicing or viewing. We use the same call signature
        as __new__ for documentation purposes. Anything not in the call
        signature is stored in annotations
        """
        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        NeoArray.__init__(self, name=name, file_origin=file_origin,
                          description=description, **annotations)

    def __repr__(self):
        """Return a string representing the :class:`AnalogSignal`."""
        return ('<%s(%s, [%s, %s], sampling rate: %s)>' %
                (self.__class__.__name__,
                 super(AnalogSignal, self).__repr__(), self.t_start,
                 self.t_stop, self.sampling_rate))

    def __getslice__(self, i, j):
        """
        Get a slice from :attr:`i` to :attr:`j`.

        Doesn't get called in Python 3, :meth:`__getitem__` is called instead
        """
        obj = super(AnalogSignal, self).__getslice__(i, j)
        obj.t_start = self.t_start + i * self.sampling_period
        return obj

    def __getitem__(self, i):
        """Get the item or slice :attr:`i`."""
        obj = super(AnalogSignal, self).__getitem__(i)
        if isinstance(obj, AnalogSignal):
            # update t_start and sampling_rate
            slice_start = None
            slice_step = None
            if isinstance(i, slice):
                slice_start = i.start
                slice_step = i.step
            elif isinstance(i, tuple) and len(i) == 2:
                slice_start = i[0].start
                slice_step = i[0].step
            if slice_start:
                obj.t_start = self.t_start + slice_start * self.sampling_period
            if slice_step:
                obj.sampling_period *= slice_step
        return obj

    # sampling_rate attribute is handled as a property so type checking can
    # be done
    @property
    def sampling_rate(self):
        """
        Number of samples per unit time.

        (1/:attr:`sampling_period`)
        """
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, rate):
        """Setter for :attr:`sampling_rate`."""
        if rate is None:
            raise ValueError('sampling_rate cannot be None')
        elif not hasattr(rate, 'units'):
            raise ValueError('sampling_rate must have units')
        self._sampling_rate = rate

    # sampling_period attribute is handled as a property on underlying rate
    @property
    def sampling_period(self):
        """Interval between two samples.

        (1/:attr:`sampling_rate`)
        """
        return 1. / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        """Setter for :attr:`sampling_period`."""
        if period is None:
            raise ValueError('sampling_period cannot be None')
        elif not hasattr(period, 'units'):
            raise ValueError('sampling_period must have units')
        self.sampling_rate = 1. / period

    # t_start attribute is handled as a property so type checking can be done
    @property
    def t_start(self):
        """Time when signal begins."""
        return self._t_start

    @t_start.setter
    def t_start(self, start):
        """Setter for :attr:`t_start`."""
        if start is None:
            raise ValueError('t_start cannot be None')
        self._t_start = start

    @property
    def duration(self):
        """Signal duration.

        (:attr:`size` * :attr:`sampling_period`)
        """
        return self.shape[0] / self.sampling_rate

    @property
    def t_stop(self):
        """Time when signal ends.

        (:attr:`t_start` + :attr:`duration`)
        """
        return self.t_start + self.duration

    @property
    def times(self):
        """The time points of each sample of the signal.

        (:attr:`t_start` + arange(:attr:`shape`)/:attr:`sampling_rate`)
        """
        return self.t_start + np.arange(self.shape[0]) / self.sampling_rate
