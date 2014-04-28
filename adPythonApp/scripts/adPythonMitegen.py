#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin
from adPythonMorph import Morph
import logging, cv2
import numpy as np
import random

class Mitegen(AdPythonPlugin):
    def __init__(self):
        # The default logging level is INFO.
        # Comment this line to set debug logging off
        self.log.setLevel(logging.DEBUG) 
        # Make some generic parameters
        # You can change the Name fields on the EDM screen here
        # Hide them by making their name -1
        params = dict(
                      # Morphology
                      m_operation = 3,
                      m_ksize = 3,    
                      m_iters = 1,                      
                      # Threshold
                      t_ksize = 11,
                      t_c = 5,
                      # Contours for cross finding
                      curve_epsilon = 8,
                      w_min = 10,
                      w_max = 100,
                      ar = 1.5,
                      ar_err = 0.15,
                      # Canny        
                      canny_thresh = 75,
                      # Output
                      step = 8,                      
                      ltype = -1,
                      lsize = -1, 
                      micron_pix = 3.67, # 5.13 for LD_*                      
                      )
        # import a morphology plugin to do filtering
        self.morph = Morph()        
        AdPythonPlugin.__init__(self, params)
        
    def paramChanged(self):
        # one of our input parameters has changed
        # pass the morph ones to the morph plugin
        for p in self.morph:
            self.morph[p] = self["m_"+p]
        self.morph.paramChanged()

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

    def get_dots(self, arr, canny, coords):        
        cx, cy, cw, ch = self.cross_params
        sums = {}
        pts = {}
        for i, coord in enumerate(coords):
            # get dims of ROI of cross image
            x = int(cx + coord[0] * (cw + 8) * np.cos(self.angle) / 5. + coord[1] * (ch + 8) * np.sin(self.angle) / 3. - 2)
            y = int(cy + coord[0] * (cw + 8) * np.sin(self.angle) / 5. + coord[1] * (ch + 8) * np.cos(self.angle) / 3. - 1)            
            w = int(cw / 5. - 4)
            h = int(ch / 3. - 5)
            # get sum of it
            sums[i] = canny[y:y+w, x:x+w].sum() / 255            
            pts[i] = (x, y, w, h)
            print coord, sums[i]            
        thresh = 0.7 * max(sums.values())
        print thresh
        tot = 0            
        for i, s in sums.items():
            x, y, w, h = pts[i]
            if self["step"] == 6:
                cv2.rectangle(arr, (x, y), (x + w, y + h), (255, 0, 0))              
            # if sum > thresh then draw a dot
            if s > thresh:
                tot += 2**i
                if self["step"] == 6:
                    cv2.circle(arr, (x + w/2, y + h/2), h/2, (0, 255, 0), 2)                
        return tot

    def processArray(self, arr, attr={}):       
     
        # convert to grey
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if self["step"] <= 0: return gray       
        
        # threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self["t_ksize"]*2+1, self["t_c"])
        if self["step"] <= 1: return thresh     

        # morphological operation
        morph = self.morph.processArray(thresh)
        if self["step"] <= 2: return morph
                             
        # Find contours
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # search for cross contours
        self.cross_params = (0, 0, 1000000, 1000000)
        for i, rawcnt in enumerate(contours):
            # approximate contour with a polygon at most curve_epsilon from the real contour
            cnt = cv2.approxPolyDP(rawcnt, self["curve_epsilon"], True)
            if self["step"] == 3:
                cv2.drawContours(arr, [cnt], 0, (0, 255, 0), 1, cv2.CV_AA)            
                
            cross = self.get_cross(cnt)                
            if cross:
                # if told to draw only crosses then do that
                if self["step"] == 4:
                    cv2.drawContours(arr, [rawcnt], 0, (0, 255, 0), 1, cv2.CV_AA)            
                # store the smallest cross
                x, y, w, h = self.cross_params
                nx, ny, nw, nh = cross
                if nw*nh < w * h:
                    # found a smaller cross, use the bounds of the raw contour
                    self.cross_params = self.get_bounds(rawcnt)
                    # Calculate angle of cross
                    moment = cv2.moments(rawcnt)
                    self.angle = 0.5*np.arctan((2*moment['mu11'])/(moment['mu20']-moment['mu02']))                 
        # if we drew on the array return it
        if self["step"] <= 4:
            return arr
        # if we didn't find anything then just return
        if not self.cross_params[0]:
            print "No crosses found in image"
            return                                
            
        # now go back to morph image and do a canny edge detect on it
        canny = cv2.Canny(morph, self["canny_thresh"], 2*self["canny_thresh"], 5);
        if self["step"] <= 5: return canny
        
        # Count the number of white pixels in each dots location
        # These dots are the loop type
        self["ltype"] = self.get_dots(arr, canny, [(1,0), (1,2), (3,2), (3,0)])
        # These dots are the loop size
        lsize = self.get_dots(arr, canny, [(0,0), (0,2), (4,2), (4,0)])
        # if we drew on the array return it
        if self["step"] <= 6:
            return arr
        
        # bounds check
        if lsize >= 9:
            return
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
        if self["step"] <= 7:
            cv2.circle(arr, (lx, ly), lr, (0, 255, 0), 2)
        return arr         

if __name__=="__main__":
    Mitegen().runOffline(
        canny_thresh=200, m_operation=11, step=8, ar = (0, 2, 0.01), ar_err = (0, 0.3, 0.01), micron_pix=(3, 7, 0.01), s_thresh=(0,1,0.01))

