#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
import logging, cv2
import numpy as np
import random

class Template(AdPythonPlugin):
    def __init__(self):
        # The default logging level is INFO.
        # Comment this line to set debug logging off
        self.log.setLevel(logging.DEBUG) 
        # Make some generic parameters
        # You can change the Name fields on the EDM screen here
        # Hide them by making their name -1
        params = dict(canny_thresh = 100,
                      curve_epsilon = 2,
                      step = 6,
                      w_min = 10,
                      w_max = 100,
                      ar = 1.5,
                      ar_err = 0.15,  
                      ksize = 3,    
                      iters = 2,
                      ltype = -1,
                      lsize = -1, 
                      micron_pix = 5.13,             
                      )
        AdPythonPlugin.__init__(self, params)
        
    def paramChanged(self):
        # one of our input parameters has changed
        # just log it for now, do nothing.
        ksize = self["ksize"]
        self.element = cv2.getStructuringElement(cv2.MORPH_OPEN, (ksize, ksize))        
        self.log.debug("Parameter has been changed %s", self)

    def get_bounds(self, cnt):
        # Get the bounds of a contour
        xs, ys = cnt[...,0], cnt[...,1]
        x, y = xs.min(), ys.min()
        w, h = xs.max() - x, ys.max() - y
        return x, y, w, h        

    def get_cross(self, cnt):
        x, y, w, h = self.get_bounds(cnt)
        if w > self["w_min"] and w < self["w_max"] and \
            h > self["w_min"] / self["ar"] and h < self["w_max"] / self["ar"] and \
            abs(w / float(h) - self["ar"]) < self["ar_err"]:
            # got a contour that might be a cross
            # check that all the points are in a cross shape
            central_x = (x + w * (0.5 - self["ar_err"]), x + w * (0.5 + self["ar_err"]))
            central_y = (y + h * (0.5 - self["ar_err"]), y + h * (0.5 + self["ar_err"]))     
            for pt in cnt:
                px, py = pt[0][0], pt[0][1]
                if (px < central_x[0] or px > central_x[1]) and \
                    (py < central_y[0] or py > central_y[1]):
                    # px and py outside central part
                    return None
            # passed cross checks
            return (x, y, w, h)
        else:
            # contour doesn't match width and aspect ratio params
            return None

    def get_dot(self, cnt):
        x, y, w, h = self.get_bounds(cnt)
        cx, cy, cw, ch = self.cross_params
        if x > cx - cw * self["ar_err"] and x < cx + (1 + self["ar_err"]) * cw and \
            y > cy - ch * self["ar_err"] and y < cy + (1 + self["ar_err"]) * ch and \
            w < cw / 5. and h < ch / 3.:        
            # got a contour that might be a dot, return it
            return (x, y, w, h)
        else:
            # not a dot
            return None

    def classify_dot(self, x, y, w, h):
        cx, cy, cw, ch = self.cross_params
        # split cross into 3 x 5 segments and see which one they fall into
        xpos = int((x - cx + w / 2.) * 5. / cw)
        ypos = int((y - cy + h / 2.) * 3. / ch)   
        # These dots are the loop type
        typelist = [(1,0), (1,2), (3,2), (3,0)]
        # These dots are the loop size
        sizelist = [(0,0), (0,2), (4,2), (4,0)]
        if (xpos, ypos) in typelist:
            return 2**typelist.index((xpos, ypos)), 0
        if (xpos, ypos) in sizelist:
            return 0, 2**sizelist.index((xpos, ypos))
        return (0, 0)

    def processArray(self, arr, attr):        
        # convert to grey
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if self["step"] == 0: return gray
        # morphological close
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self.element, iterations=self["iters"])
        if self["step"] == 1: return gray        
        # Canny edge detect        
        canny_output = cv2.Canny(gray, self["canny_thresh"], 2*self["canny_thresh"], 5);
        if self["step"] == 2: return canny_output
        # Find contours
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # search for cross contours
        self.cross_params = None
        for i, rawcnt in enumerate(contours):
            # approximate contour with a polygon at most curve_epsilon from the real contour
            cnt = cv2.approxPolyDP(rawcnt, self["curve_epsilon"], True)
            cross = self.get_cross(cnt)
            if self["step"] == 3:
                cv2.drawContours(arr, contours, i, (0, 255, 0), 1, cv2.CV_AA)            
            if cross:
                self.cross_params = cross
                moment = cv2.moments(rawcnt)
                self.angle = 0.5*np.arctan((2*moment['mu11'])/(moment['mu20']-moment['mu02']))                 
                if self["step"] == 4:
                    cv2.drawContours(arr, [cnt], 0, (0, 255, 0), 1, cv2.CV_AA)
                    return arr  
                break
        # if we didn't find anything then just return
        if self.cross_params is None:
            return                        
        # now go back to the contours and look for dots
        self.dots = []
        for i, rawcnt in enumerate(contours):
            cnt = cv2.approxPolyDP(rawcnt, self["curve_epsilon"], True)
            dot = self.get_dot(cnt)
            if dot:
                self.dots.append(dot)
                if self["step"] == 5:
                    cv2.drawContours(arr, contours, i, (0, 255, 0), 1, cv2.CV_AA)
        # got a list of dots, work out binary code
        coords = set()
        for dot in self.dots:
            coords.add(self.classify_dot(*dot))
        self["ltype"], lsize = sum(np.array(x) for x in coords)
        # lookup loop size based on type
        if self["ltype"] == 4:
            # microloops
            self["lsize"] = (10, 20, 35, 50, 75, 100, 150, 200, 300)[lsize]
        elif self["ltype"] == 1:
            # M2
            self["lsize"] = (10, 20, 30, 50, 75, 100, 150, 200, 300)[lsize]            
        else:
            # not supported
            self["lsize"] = -1
            print "Loop type %d not supported" % self["ltype"]
            return
        # Now draw the loop, we assume the holder is mounted roughly level and 870 microns between loop and cross
        cx, cy, cw, ch = self.cross_params        
        loop_off = 870. / self['micron_pix']
        lx = np.int32(cx + cw / 2. - np.cos(self.angle) * loop_off + 0.5)
        ly = np.int32(cy + ch / 2. - np.sin(self.angle) * loop_off + 0.5)
        lr = np.int32(self["lsize"] / 2. / self['micron_pix'] + 0.5)
        if self["step"] == 6:
            cv2.circle(arr, (lx, ly), lr, (0, 255, 0), 2)
        return arr         

if __name__=="__main__":
    Template().runOffline(
        canny_thresh=200, step=7, ar = (0, 2, 0.01), ar_err = (0, 0.3, 0.01), micron_pix=(5, 7, 0.01))

