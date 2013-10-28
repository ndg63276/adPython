#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
import cv2
import numpy
import logging

# These are the operation types
MORPH_ERODE=0
MORPH_DILATE=1
MORPH_OPEN=2
MORPH_CLOSE=3
MORPH_GRADIENT=4
MORPH_TOPHAT=5
MORPH_BLACKHAT=6
MORPH_BLUR=7
MORPH_GAUSSIAN_BLUR=8
MORPH_MEDIAN_BLUR=9

# Set a debug logging level in the local logger
logging.getLogger(".".join([__name__, "Focus"])).setLevel(logging.DEBUG)

class Focus(AdPythonPlugin):
    def __init__(self):
        params = dict(ksize = 3, prefilter = 0, iterations = 1,
                      sum = 0.0, filtered_mean = 0.0, filtered_stddev = 0.0)
        AdPythonPlugin.__init__(self, params)
        
    def paramChanged(self):
        # one of our input parameters has changed
        ksize = self["ksize"]
        self.element = cv2.getStructuringElement(cv2.MORPH_OPEN, (ksize, ksize))
        self.log.info('Changed parameter, ksize=%s', str(ksize))

    def processArray(self, arr, attr):
        # got a new image to process
        self.log.debug("arr size: %s", arr.shape)
        self.log.debug("parameters: %s", str(self._params))
        
        if self['prefilter'] > 0:
            dest = cv2.morphologyEx(img, cv2.MORPH_OPEN, self.element)
        else:
            dest = arr
        dest = cv2.morphologyEx(dest, cv2.MORPH_GRADIENT, 
                                self.element, iterations = self['iterations'])
        #hist = numpy.histogram(dest, bins = self.params['bins'], range = (100,5000))
        #correcthist = (hist[0], hist[1][:-1])
        meanstddev = cv2.meanStdDev(dest)
        self.log.debug("mean stddev: %s", str(meanstddev))
        self['filtered_mean'] = meanstddev[0][0][0]
        self['filtered_stddev'] = meanstddev[1][0][0]
        self['sum'] = cv2.sumElems(dest)[0]
        
        return dest

if __name__=="__main__":
    Focus().runOffline()
