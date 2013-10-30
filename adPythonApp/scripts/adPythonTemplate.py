#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin

# We might need the numpy library to do array operations
import numpy, time

class Template(AdPythonPlugin):
    def __init__(self):
        # Make some generic parameters
        # You can change the Name fields on the EDM screen here
        # Hide them by making their name -1
        params = dict(int1 = 1,      int1Name = "Int 1",
                      int2 = 2,      int2Name = "Int 2",
                      int3 = 3,      int3Name = "-1",
                      double1 = 1.0, double1Name = "Double 1",
                      double2 = 2.0, double2Name = "Double 2",
                      double3 = 3.0, double3Name = "-1")
        AdPythonPlugin.__init__(self, params)
        
    def paramChanged(self):
        # one of our input parameters has changed
        # just log it for now, do nothing
        self.log.info("Hello world")

    def processArray(self, arr, attr):
        # got a new image to process, lets add int1 to it and return it
        return arr + self["int1"]

if __name__=="__main__":
    Template().runOffline()
