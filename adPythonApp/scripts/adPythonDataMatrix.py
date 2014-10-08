#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
import logging, numpy

import cv2
from pydmtx import DataMatrix

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return numpy.dot(d1, d2) / numpy.sqrt( numpy.dot(d1, d1)*numpy.dot(d2, d2) )

def dist_between(p0, p1):
    d = (p0-p1).astype('float') 
    return numpy.sqrt(numpy.dot(d, d))

class Template(AdPythonPlugin):
    def __init__(self):
        # The default logging level is INFO.
        # Comment this line to set debug logging off
        self.log.setLevel(logging.DEBUG) 
        # Make some generic parameters
        # You can change the Name fields on the EDM screen here
        # Hide them by making their name -1
        params = dict(int1 = 101,      int1Name = "Threshold block size",
                      int2 = 13,      int2Name = "Threshold C",
                      int3 = 3,      int3Name = "Kernel size",
                      double1 = 0.35, double1Name = "Angle cos",
                      double2 = 6.0, double2Name = "Curve epsilon",
                      double3 = 15.0, double3Name = "Length diff")
        self.dm=DataMatrix()
        AdPythonPlugin.__init__(self, params)
        
    def paramChanged(self):
        # one of our input parameters has changed
        # just log it for now, do nothing.
        self.log.debug("Parameter has been changed %s", self)

    def processArray(self, arr, attr={}):
        # Turn the picture gray
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        # copy the array output
        out = arr.copy()
        # Do an adaptive threshold to get a black and white image which factors in lighting
        thresh = cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self["int1"], self["int2"])
        # Morphological filter to get rid of noise
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self["int3"], self["int3"]))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element, iterations=1)      
        m = morph.copy()  
        # Find squares in the image
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        squares = []
        for rawcnt in contours:
            cnt = cv2.approxPolyDP(rawcnt, self["double2"], True)
            cnt = cnt.reshape(-1, 2)
            l = len(cnt)
            if l < 4:
                continue
            # find distances between successive points
            dists = [(dist_between(cnt[i], cnt[(i+1)%l]),i) for i in xrange(l)]
            dists.sort(reverse=True)            
            # if longest 2 line lengths are about the same and segments are 
            # next to each other           
            (d0, i0), (d1, i1) = dists[0], dists[1]
            if abs(d0 - d1) < self["double3"] and abs(i0 - i1) in (1, l-1) and \
                d0 > self["int1"] and d1 > self["int1"]:
                # Find out which point is first in contour
                if i0 < i1 or (i1 == 0 and i0 == l-1):
                    first = i0
                else:
                    first = i1
                # Get cos of the angle of the corner                
                pts = [cnt[first], cnt[(first+1)%l], cnt[(first+2)%l]]
                ac = angle_cos(*pts)
                if abs(ac) < self["double1"]:                      
                    # work out the rotation of the lines
                    angle0 = numpy.degrees(numpy.arctan((pts[0][0] - pts[1][0])/float(pts[1][1]-pts[0][1])))
                    angle2 = numpy.degrees(numpy.arctan((pts[2][0] - pts[1][0])/float(pts[1][1]-pts[2][1])))
                    # rotate vector by 90 degrees clockwise to get last point
                    dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
                    p3 = [pts[0][0] - dy, pts[0][1] + dx]
                    pts.append(p3)
                    pts = numpy.array(pts, 'int32')
                    # now work out the bounding rectangle of those points
                    x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
                    # and take a roi of it
                    pad = self["int1"] / 10
                    threshroi = thresh[y-pad:y+h+pad, x-pad:x+w+pad]
                    # work out the rotated bounding rectangle
                    center, size, angle = cv2.minAreaRect(pts-(x-pad, y-pad))                
                    if angle0 > angle2:
                        if pts[1][1]-pts[0][1] < 0:
                            angle += 180
                    else:
                        if pts[1][1]-pts[0][1] < 0:
                            angle += 90
                        else:
                            angle += 270  
                    # get the rotation matrix                            
                    M = cv2.getRotationMatrix2D(center, angle, 1.0);
                    # perform affine transform
                    l = max(*threshroi.shape)
                    rot = cv2.warpAffine(threshroi, M, (l, l));
                    # now decode it
                    sym = self.dm.decode(l, l, buffer(rot.tostring()), max_count = 1)
                    print sym
                    if sym:
                        cv2.putText(out, sym, (int(center[0]+x+pad), int(center[1]+y+pad)), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
                        squares.append(pts)                        
                    
        # Draw squares on the image    
        cv2.drawContours(out, squares, -1, (255,0,0), 3)
        
        # Mask out the area of these polys   
        #cv2.imwrite("/scratch/U/datamatrix_results.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR) )
        return out

if __name__=="__main__":
    Template().runOffline(double1=(0, 1, 0.01), double3=(0,300))

