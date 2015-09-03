#!/usr/bin/env dls-python
from adPythonPlugin import AdPythonPlugin

import cv2


# Use thin wrapper functions around cv2 operations because we want a unified
# interface (fn(arr, params)) to more easily expose the functions to the user.

def erode(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(arr, element, iterations=iterations)


def dilate(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(arr, element, iterations=iterations)


# `_morph` suffix to avoid name collision.
def open_morph(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(
        arr, cv2.MORPH_OPEN, element, iterations=iterations)


def close(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(
        arr, cv2.MORPH_CLOSE, element, iterations=iterations)


def gradient(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(
        arr, cv2.MORPH_GRADIENT, element, iterations=iterations)


def top_hat(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(
        arr, cv2.MORPH_TOPHAT, element, iterations=iterations)


def black_hat(arr, params):
    ksize, iterations = params[:2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(
        arr, cv2.MORPH_BLACKHAT, element, iterations=iterations)


def blur(arr, params):
    # The comma is necessary. (It unpacks the tuple.)
    ksize, = params[:1]
    return cv2.blur(arr, (ksize, ksize))


def gaussian_blur(arr, params):
    ksize, = params[:1]
    # Kernel size should be odd.
    if not ksize % 2:
        ksize += 1
    return cv2.GaussianBlur(arr, (ksize, ksize), 0)


def median_blur(arr, params):
    ksize, = params[:1]
    if not ksize % 2:
        ksize += 1
    return cv2.medianBlur(arr, ksize)
    

def canny_edge_detect(arr, params):
    upper_threshold, lower_threshold = params[:2]

    # Upper and lower threshold arguments commute.
    return cv2.Canny(arr, upper_threshold, lower_threshold)


# List of candidate preprocessing functions.
# Order must match that in mbb* records.
pp_candidates = [
    erode,
    dilate,
    open_morph,
    close,
    gradient,
    top_hat,
    black_hat,
    blur,
    gaussian_blur,
    median_blur,
    lambda arr, params: arr,  # The "identity" process.
]


def locate_sample(edge_arr, params):
    # Straight port of Tom Cobb's algorithm from the original (adOpenCV) 
    # mxSampleDetect.

    direction, min_tip_height = params[:2]

    # Index into edges_arr like [y, x], not [x, y]!
    height, width = edge_arr.shape

    tip_y, tip_x = None, None
    top = [None]*width
    bottom = [None]*width

    rows = xrange(height)
    if direction == 1:
        columns = xrange(width)
    else:
        assert direction == -1
        columns = reversed(xrange(width))

    for x in columns:
        for y in rows:

            if not edge_arr[y, x]:
                continue
                
            if top[x] is None:
                top[x] = y

            bottom[x] = y

        # Look for the first non-narrow region between top and bottom edges.
        if tip_x is None and top[x] is not None \
        and abs(top[x] - bottom[x]) > min_tip_height:
            
            # Move backwards to where there were no edges at all...
            while top[x] is not None:
                x += -direction
                if x == -1 or x == width:
                    # (In this case the sample is off the edge of the picture.)
                    break
            x += direction # ...and forward one step. x is now at the tip.

            tip_x = x
            tip_y = int(round(0.5*(top[x] + bottom[x])))

            # Zero the edge arrays to the left (right) of the tip.
            if direction == 1:
                top[:x] = [None for _ in xrange(x)]
                bottom[:x] = [None for _ in xrange(x)]
            else:
                assert direction == -1
                top[x:] = [None for _ in xrange(x)]
                bottom[x:] = [None for _ in xrange(x)]

    if tip_y is None or tip_x is None:
        tip_y, tip_x = -1, -1

    return (tip_y, tip_x), (top, bottom)


class MxSampleDetect(AdPythonPlugin):

    def __init__(self):

        # Default values. All params are integers in this case.
        params = dict(
            preprocess=0,  # Choose from the list of candidate functions.
            pp_param1=3,  # Generic parameter for preprocessing.
            pp_param2=1,  # Another. (Meaning imbued by use.)
            canny_upper=100,  # Thresholds for Canny edge detection.
            canny_lower=50,
            close_ksize=5,  # Kernel size for "close" operation.
            close_iterations=1,
            scan_direction=1,  # +1:LtR, -1:RtL
            min_tip_height=5,
            tip_x=-1,  # Pixel positions of detected tip.
            tip_y=-1,  # (Not really parameters...)
            out_arr=0,  # Which array to put downstream.
        )

        AdPythonPlugin.__init__(self, params)

    def processArray(self, arr, attr={}):
        # Get a greyscale version of the input.
        dimensions = len(arr.shape)
        if dimensions == 3 or dimensions == 4:
            gray_arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            assert dimensions == 2
            gray_arr = arr

        # Preprocess the array. (Use the greyscale one.)
        pp_params = (self['pp_param1'], self['pp_param2'])
        pp_arr = pp_candidates[self['preprocess']](gray_arr, pp_params)

        # (Could do a remove_dirt step here if wanted.)

        # Find some edges.
        canny_params = (self['canny_upper'], self['canny_lower'])
        edge_arr = canny_edge_detect(pp_arr, canny_params)

        # Do a "close" image operation. (Add other options?)
        close_params = (self['close_ksize'], self['close_iterations'])
        closed_arr = close(edge_arr, close_params)

        # Find the sample.
        location_params = (self['scan_direction'], self['min_tip_height'])
        tip, edges = locate_sample(closed_arr, location_params)

        # Write our results to PVs.
        self['tip_y'], self['tip_x'] = tip
        # TODO: Write top, bottom edge arrays to PVs.

        # Return whichever array the user wants passed down to others in the
        # image processing chain.
        return (arr, gray_arr, pp_arr, edge_arr, closed_arr)[self['out_arr']]


if __name__ == '__main__':
    # This script can be run offline with
    # `PYTHONPATH+=../src/ dls-python adPythonMxSampleDetect.py`.

    # Args passed to .runOffline define ranges for sliders.
    MxSampleDetect().runOffline(
        preprocess=11,
        pp_param1=200,
        pp_param2=200,
        canny_upper=256,
        canny_lower=256,
        close_ksize=200,
        close_iterations=200,
        min_tip_height=100,
        tip_x=500,
        tip_y=500,
        out_arr=5,
    )
