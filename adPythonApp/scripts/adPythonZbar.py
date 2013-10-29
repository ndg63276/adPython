#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
import cv2
import numpy
import sys

sys.path.append('/dls_sw/work/tools/RHEL6-x86_64/zbar/prefix/lib/python2.7/site-packages')
import zbar
import logging

class Zbar(AdPythonPlugin):
    def __init__(self):
        # turn on debugging just for this class
        self.log.setLevel(logging.DEBUG)
        params = dict(data = "", type = "", count = 0, quality = 0, busy = 0)
        AdPythonPlugin.__init__(self, params)
        
        self.scanner = zbar.ImageScanner()
        self.scanner.parse_config('enable')
        self._busy = 0
        self._data_latch = ""
        
    def paramChanged(self):
        # Check if user has made busy record busy
        if self['busy'] == 1 and (self['busy'] != self._busy):
            self.log.debug('Starting scan!')
            self._busy = 1
            self['count'] = 0
            self['data'] = ""
            self['type'] = ""
            self['quality'] = 0

    def processArray(self, arr, attr):
        # got a new image to process
        self.log.debug("arr size: %s", arr.shape)
        
        # Create a zbar image wrapper around the raw data from the array
        # Assumption here is that the array is a 2D, 8bpp greyscale image.
        # TODO: support other image types
        zimg = zbar.Image( arr.shape[0], arr.shape[1], 'Y800', arr.tostring() )
        
        # Scan image wrapper for barcodes. Results are attached to the zimg object
        self.scanner.scan(zimg)
        
        symbol = None
        for symbol in zimg:
            self.log.debug("type: %6s    quality: %d     data: %s", \
                           symbol.type, symbol.quality, symbol.data )
            #self.log.debug("Locations: %s", str(symbol.location))

        dest = arr        
        if symbol != None:
            # Only update results if user is waiting for a result or if a new barcode
            # has been spotted.
            self.log.debug('_busy=%d   _data_latch=%s', self._busy, self._data_latch)
            if (self._busy == 1) or (symbol.data != self._data_latch and symbol.data != ""):
                self._data_latch = symbol.data
                self['count'] = symbol.count
                self['data'] = symbol.data
                self['type'] = str(symbol.type)
                self['quality'] = symbol.quality
                # clear the busy flag
                self._busy = 0
                self['busy'] = 0

                #points = numpy.array(symbol.location, numpy.int32)
                #polygons = points.reshape((-1, 1, 2))
                #self.log.debug("Polygons: %s", str(polygons))
                #dest = cv2.polylines(arr, [polygons], True, 255)
                dest = arr

                self.log.info("results: %s", str(self._params))
        #self.log.debug("dest: %s", str(dest))
        return dest

if __name__=="__main__":
    Focus().runOffline()
