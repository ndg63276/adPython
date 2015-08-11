#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
import logging, numpy


class PowerMean(AdPythonPlugin):
    '''Computes the mean power of a sequence of updates.'''

    def __init__(self):
        # The default logging level is INFO.
        # Comment this line to set debug logging off
        self.log.setLevel(logging.DEBUG)

        # The only parameter is the sample count for averaging.
        params = dict(count = 1, countName = 'Number of samples to average')
        AdPythonPlugin.__init__(self, params)

        self.seen = 0

    def paramChanged(self):
        # one of our input parameters has changed
        # just log it for now, do nothing.
        self.log.debug('Parameter has been changed %s', self)

    def processArray(self, arr, attr={}):
        if self.seen:
            self.data += arr * arr
            self.seen += 1
        else:
            self.data = arr * arr
            self.seen = 1

        if self.seen >= self.count:
            result = numpy.sqrt(self.data / self.seen)
            self.seen = 0
            self.data = None
            return result
        else:
            # No value yet
            return None
