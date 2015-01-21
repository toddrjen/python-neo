# -*- coding: utf-8 -*-
"""This module implements :class:`SpikeTrain`, an array of spike times.

:class:`SpikeTrain` derives from :class:`NeoArray`, from
:module:`neo.core.neoarray`.
"""

# needed for python 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq

from neo.core.neoarray import NeoArray, _check_time_in_range


def _new_spiketrain(cls, signal, t_stop, units=None, dtype=None,
                    copy=True, sampling_rate=1.0 * pq.Hz,
                    t_start=0.0 * pq.s, waveforms=None, left_sweep=None,
                    name=None, file_origin=None, description=None,
                    annotations=None):
    """Map SpikeTrain.__new__ to function that does not do the unit checking.

    This is needed for pickle to work.
    """
    if annotations is None:
        annotations = {}
    return SpikeTrain(signal, t_stop, units, dtype, copy, sampling_rate,
                      t_start, waveforms, left_sweep, name, file_origin,
                      description, **annotations)


class SpikeTrain(NeoArray):

    """:class:`SpikeTrain` is a :class:`Quantity` array of spike times.

    It is an ensemble of action potentials (spikes) emitted by the same unit
    in a period of time.

    *Usage*::

        >>> from neo.core import SpikeTrain
        >>> from quantities import s
        >>>
        >>> train = SpikeTrain([3, 4, 5]*s, t_stop=10.0)
        >>> train2 = train[1:3]
        >>>
        >>> train.t_start
        array(0.0) * s
        >>> train.t_stop
        array(10.0) * s
        >>> train
        <SpikeTrain(array([ 3.,  4.,  5.]) * s, [0.0 s, 10.0 s])>
        >>> train2
        <SpikeTrain(array([ 4.,  5.]) * s, [0.0 s, 10.0 s])>


    *Required attributes/properties*:
        :times: (quantity array 1D, numpy array 1D, or list) The times of
            each spike.
        :units: (quantity units) Required if :attr:`times` is a list or
                :class:`~numpy.ndarray`, not if it is a
                :class:`~quantites.Quantity`.
        :t_stop: (quantity scalar, numpy scalar, or float) Time at which
            :class:`SpikeTrain` ended. This will be converted to the
            same units as :attr:`times`. This argument is required because it
            specifies the period of time over which spikes could have occurred.
            Note that :attr:`t_start` is highly recommended for the same
            reason.

    Note: If :attr:`times` contains values outside of the
    range [t_start, t_stop], an Exception is raised.

    *Recommended attributes/properties*:
        :name: (str) A label for the dataset.
        :description: (str) Text description.
        :file_origin: (str) Filesystem path or URL of the original data file.
        :t_start: (quantity scalar, numpy scalar, or float) Time at which
            :class:`SpikeTrain` began. This will be converted to the
            same units as :attr:`times`.
            Default: 0.0 seconds.
        :waveforms: (quantity array 3D (spike, channel_index, time))
            The waveforms of each spike.
        :sampling_rate: (quantity scalar) Number of samples per unit time
            for the waveforms.
        :left_sweep: (quantity array 1D) Time from the beginning
            of the waveform to the trigger time of the spike.
        :sort: (bool) If True, the spike train will be sorted by time.

    *Optional attributes/properties*:
        :dtype: (numpy dtype or str) Override the dtype of the signal array.
        :copy: (bool) Whether to copy the times array.  True by default.
            Must be True when you request a change of units or dtype.

    Note: Any other additional arguments are assumed to be user-specific
            metadata and stored in :attr:`annotations`.

    *Properties available on this object*:
        :sampling_period: (quantity scalar) Interval between two samples.
            (1/:attr:`sampling_rate`)
        :duration: (quantity scalar) Duration over which spikes can occur,
            read-only.
            (:attr:`t_stop` - :attr:`t_start`)
        :spike_duration: (quantity scalar) Duration of a waveform, read-only.
            (:attr:`waveform`.shape[2] * :attr:`sampling_period`)
        :right_sweep: (quantity scalar) Time from the trigger times of the
            spikes to the end of the waveforms, read-only.
            (:attr:`left_sweep` + :attr:`spike_duration`)
        :times: (:class:`SpikeTrain`) Returns the :class:`SpikeTrain` without
            modification or copying.

    *Slicing*:
        :class:`SpikeTrain` objects can be sliced. When this occurs, a new
        :class:`SpikeTrain` (actually a view) is returned, with the same
        metadata, except that :attr:`waveforms` is also sliced in the same way
        (along dimension 0). Note that t_start and t_stop are not changed
        automatically, although you can still manually change them.

    """

    _single_parent_objects = ('Segment', 'Unit')
    _main_attr = 'times'
    _necessary_attrs = (('t_start', pq.Quantity, 0),
                        ('t_stop', pq.Quantity, 0))
    _recommended_attrs = ((('waveforms', pq.Quantity, 3),
                           ('left_sweep', pq.Quantity, 0),
                           ('sampling_rate', pq.Quantity, 0)) +
                          NeoArray._recommended_attrs)
    _ndarray_slice_attrs = ('waveforms',)
    _required_dimensionality = pq.UnitTime
    _new_wrapper = _new_spiketrain
    _default_vals = {'_t_start': 0 * pq.s}

    def __new__(cls, times, t_stop, units=None, dtype=None, copy=True,
                sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s, waveforms=None,
                left_sweep=None, name=None, file_origin=None, description=None,
                **annotations):
        """Construct new :class:`SpikeTrain` from data.

        This is called whenever a new class:`SpikeTrain` is created from
        the constructor, but not when slicing.

        __array_finalize__ is called on the new object.
        """
        # Construct Quantity from data
        obj = NeoArray.__new__(cls, data=times, units=units, dtype=dtype,
                               copy=copy, name=name, file_origin=file_origin,
                               description=description, **annotations)

        if dtype is not None and hasattr(times.dtype) and times.dtype != dtype:
            # if t_start.dtype or t_stop.dtype != times.dtype != dtype,
            # _check_time_in_range can have problems, so we set the t_start
            # and t_stop dtypes to be the same as times before converting them
            # to dtype below
            # see ticket #38
            if hasattr(t_start, 'dtype') and t_start.dtype != obj.dtype:
                t_start = t_start.astype(obj.dtype)
            if hasattr(t_stop, 'dtype') and t_stop.dtype != obj.dtype:
                t_stop = t_stop.astype(obj.dtype)

        # if the dtype and units match, just copy the values here instead
        # of doing the much more epxensive creation of a new Quantity
        # using items() is orders of magnitude faster
        dim = obj.units.dimensionality
        if (hasattr(t_start, 'dtype') and t_start.dtype == obj.dtype and
                hasattr(t_start, 'dimensionality') and
                t_start.dimensionality.items() == dim.items()):
            obj.t_start = t_start.copy()
        else:
            obj.t_start = pq.Quantity(t_start, units=dim, dtype=dtype)

        if (hasattr(t_stop, 'dtype') and t_stop.dtype == obj.dtype and
                hasattr(t_stop, 'dimensionality') and
                t_stop.dimensionality.items() == dim.items()):
            obj.t_stop = t_stop.copy()
        else:
            obj.t_stop = pq.Quantity(t_stop, units=dim, dtype=dtype)

        # Store attributes
        obj.waveforms = waveforms
        obj.left_sweep = left_sweep
        obj.sampling_rate = sampling_rate

        # Error checking (do earlier?)
        _check_time_in_range(obj, obj.t_start, obj.t_stop, view=True)

        return obj

    def __init__(self, times, t_stop, units=None,  dtype=np.float,
                 copy=True, sampling_rate=1.0 * pq.Hz, t_start=0.0 * pq.s,
                 waveforms=None, left_sweep=None, name=None, file_origin=None,
                 description=None, **annotations):
        """Initialize a newly constructed :class:`SpikeTrain` instance.

        This method is only called when constructing a new SpikeTrain,
        not when slicing or viewing. We use the same call signature
        as __new__ for documentation purposes. Anything not in the call
        signature is stored in annotations.
        """
        # Calls parent __init__, which grabs universally recommended
        # attributes and sets up self.annotations
        NeoArray.__init__(self, data=times, units=units, dtype=dtype,
                          copy=copy, name=name, file_origin=file_origin,
                          description=description, **annotations)

    @property
    def _new_wrapper_args(self):
        """Return a tuple of the arguments used to construct _new_wrapper."""
        #      (times, t_stop, units, dtype, copy,
        #       sampling_rate, t_start, waveforms,
        #       left_sweep, name, file_origin,
        #       description, **anotations)
        return (np.array(self), self.t_stop, self.units, self.dtype, True,
                self.sampling_rate, self.t_start, self.waveforms,
                self.left_sweep,  self.name, self.file_origin,
                self.description, self.annotations)

    def __repr__(self):
        """Return a string representing the :class:`SpikeTrain`."""
        return '<SpikeTrain(%s, [%s, %s])>' % (
            super(SpikeTrain, self).__repr__(), self.t_start, self.t_stop)

    def __setitem__(self, i, value):
        """Set the value of the item or slice :attr:`i`."""
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
            # or should we be strict: raise ValueError("Setting a value
            # requires a quantity")?
        # check for values outside t_start, t_stop
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setitem__(i, value)

    def __setslice__(self, i, j, value):
        """Set the value of the slice :attr:`i`.

        Doesn't get called in Python 3, :meth:`__setitem__` is called instead
        """
        if not hasattr(value, "units"):
            value = pq.Quantity(value, units=self.units)
        _check_time_in_range(value, self.t_start, self.t_stop)
        super(SpikeTrain, self).__setslice__(i, j, value)

    def time_slice(self, t_start, t_stop):
        """Slice the :class:`SpikeTrain` by time.

        Creates a new :class:`SpikeTrain` corresponding to the time slice of
        the original :class:`SpikeTrain` between (and including) times
        :attr:`t_start` and :attr:`t_stop`. Either parameter can also be None
        to use infinite endpoints for the time interval.
        """
        _t_start = t_start
        _t_stop = t_stop
        if t_start is None:
            _t_start = -np.inf
        if t_stop is None:
            _t_stop = np.inf
        indices = (self >= _t_start) & (self <= _t_stop)
        new_st = self[indices]

        new_st.t_start = max(_t_start, self.t_start)
        new_st.t_stop = min(_t_stop, self.t_stop)
        if self.waveforms is not None:
            new_st.waveforms = self.waveforms[indices]

        return new_st

    @property
    def times(self):
        """Return the :class:`SpikeTrain` without modification or copying."""
        return self

    @property
    def duration(self):
        """Duration over which spikes can occur.

        (:attr:`t_stop` - :attr:`t_start`)
        """
        if self.t_stop is None or self.t_start is None:
            return None
        return self.t_stop - self.t_start

    @property
    def spike_duration(self):
        """Duration of a waveform.

        (:attr:`waveform`.shape[2] * :attr:`sampling_period`)
        """
        if self.waveforms is None or self.sampling_rate is None:
            return None
        return self.waveforms.shape[2] / self.sampling_rate

    @property
    def sampling_period(self):
        """Interval between two samples.

        (1/:attr:`sampling_rate`)
        """
        if self.sampling_rate is None:
            return None
        return 1.0 / self.sampling_rate

    @sampling_period.setter
    def sampling_period(self, period):
        """Setter for :attr:`sampling_period`."""
        if period is None:
            self.sampling_rate = None
        else:
            self.sampling_rate = 1.0 / period

    @property
    def right_sweep(self):
        """Time from the trigger times to the end of the waveforms.

        (:attr:`left_sweep` + :attr:`spike_duration`)
        """
        dur = self.spike_duration
        if self.left_sweep is None or dur is None:
            return None
        return self.left_sweep + dur
