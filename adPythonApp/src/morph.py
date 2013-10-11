#!/usr/bin/env python
import cv2
from adPythonPlugin import AdPythonPlugin

class morph(AdPythonBase):
    print "making class"
    def __init__(self, ptr=None):
        params = dict(ksize = 3, iters = 1, zoo=32.)
        AdPythonBase.__init__(self, ptr, params)
        
    def paramChanged(self):
        # one of our input parameters has changed
        ksize = self["ksize"]
        self.element = cv2.getStructuringElement(cv2.MORPH_OPEN, (ksize, ksize))
        
    def processArray(self, arr, attr):
        # got a new image to process
        dest = cv2.morphologyEx(arr, cv2.MORPH_ELLIPSE, self.element, iterations=self["iters"])
        print dest[0][0:4]
        return dest

if __name__=="__main__":
    morph().runOffline()
