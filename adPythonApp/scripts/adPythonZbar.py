#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
import cv2
import numpy
import sys
import logging

sys.path.append('/dls_sw/work/tools/RHEL6-x86_64/zbar/prefix/lib/python2.7/site-packages')
import zbar

sys.path.append('/dls_sw/work/tools/RHEL6-x86_64/pydmtx/prefix/lib/python2.7/site-packages')
import pydmtx

class BarCodeSymbol:
    def __init__(self):
        self.data = ''
        self.type = ''
        self.polygon = []
        self.quality = 0
        
    def __str__(self):
        s = "<BarCodeSymbol type=\'%s\', data=\'%s\', quality=%d >" % (self.type, self.data, self.quality)
        return s
    
    def points_to_polygon(self, points):
        '''Takes a set of points in lists or tuples: ((x1,y1), (x2,y2), ... (xn,yn))
        and converts it into a shape that OpenCV can treat as points in a polygon.'''
        points = numpy.array(points, numpy.int32)
        self.polygon = points.reshape((-1, 1, 2))
        
class BarCodeDecoder:
    '''BarDecoder base class to provide a common interface for 
    implementing various barcode libraries decode support'''
    def __init__(self, logger='BarCodeDecoder'):
        self.log = logging.getLogger(logger)
        self.symbols = []
        
    def __str__(self):
        s = "< BarDecoder symbols: %s >" % str(self.symbols)
        return s
        
    def decode(self, array):
        '''Decode the barcode symbols from a 2D array
        The input array should be a numpy 2D array of unsigned 8 bpp elements.
        Returns the number of symbols decoded from the array.'''
        raise NotImplementedError

class ZBarDecoder(BarCodeDecoder):
    '''Use ZBar to decode a range of barcode types'''
    def __init__(self, logger='ZBarDecoder'):
        BarCodeDecoder.__init__(self, logger)
        self._scanner = zbar.ImageScanner()
        self._scanner.parse_config('enable')
    
    def decode(self, array):
        self.symbols = []
        zimg = zbar.Image( array.shape[0], array.shape[1], 'Y800', array.tostring() )
        self._scanner.scan(zimg)
        for zb_symbol in zimg:
            symbol = BarCodeSymbol()
            symbol.type = str(zb_symbol.type)
            symbol.data = zb_symbol.data
            symbol.quality = int(zb_symbol.quality)
            symbol.points_to_polygon(zb_symbol.location)
            self.symbols.append(symbol)
        return len(self.symbols)

class DmtxDecoder(BarCodeDecoder):
    '''Use the pydmtx - python bindings for libdmtx to decode Data Matrix barcodes'''
    def __init__(self, logger='DmtxDecoder'):
        BarCodeDecoder.__init__(self, logger)
        self._dm_read = pydmtx.DataMatrix()
        
    def decode(self, array):
        self.symbols = []
        self._dm_read.decode( array.shape[0], array.shape[1], buffer(array))
        num_hits = self._dm_read.count()
        for i in range(num_hits):
            symbol = BarCodeSymbol()
            symbol.data = str(self._dm_read.message(i))
            symbol.type = str(self._dm_read.stats(i))
            self.symbols.append(symbol)
        return len(self.symbols)

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
        #self.log.debug("arr size: %s", arr.shape)
        
        # Create a zbar image wrapper around the raw data from the array
        # Assumption here is that the array is a 2D, 8bpp greyscale image.
        # TODO: support other image types
        zimg = zbar.Image( arr.shape[0], arr.shape[1], 'Y800', arr.tostring() )
        
        # Scan image wrapper for barcodes. Results are attached to the zimg object
        self.scanner.scan(zimg)
        
        symbol = None
        count = 0
        for symbol in zimg:
            count += 1
            self.log.debug("type: %6s    quality: %d     data: %s", \
                           symbol.type, symbol.quality, symbol.data )
            self.log.debug("Locations: %s", str(symbol.location))

        dest = None
        if symbol != None:
            # Only update results if user is waiting for a result or if a new barcode
            # has been spotted.
            self.log.debug('_busy=%d   _data_latch=%s', self._busy, self._data_latch)
            if (self._busy == 1) or (symbol.data != self._data_latch and symbol.data != ""):
                self._data_latch = symbol.data
                self['count'] = count
                self['data'] = symbol.data
                self['type'] = str(symbol.type)
                self['quality'] = symbol.quality
                # clear the busy flag
                self._busy = 0
                self['busy'] = 0

                points = numpy.array(symbol.location, numpy.int32)
                polygons = points.reshape((-1, 1, 2))
                #self.log.debug("Polygons: %s", str(polygons))
                dest = numpy.array(arr)
                cv2.polylines(dest, [polygons], True, 0, 5)
                self.log.info("Drawing. Output: %s", str(dest.shape))
        #self.log.debug("dest: %s", str(dest))
        return dest

if __name__=="__main__":
    Focus().runOffline()
