class AdPythonPlugin(object):   
    # Will be our param dict
    _params = None
    
    # init our param dict
    def __init__(self, params={}):
        self._params = dict(params)
        self.paramChanged()

    # get a param value
    def __getitem__(self, param):
        return self._params[param]

    # set a param value 
    def __setitem__(self, param, value):
        assert param in self, "Param %s not in param lib" % param
        self._params[param] = value
        self.paramChanged()
 
    # see if param is supported
    def __contains__(self, param):
        return param in self._params
 
    # length of param dict
    def __len__(self):
        return len(self._params)

    # for if we want to print the dict 
    def __repr__(self):
        return repr(self._params)

    # called when parameter list changes
    def paramChanged(self):
        raise NotImplementedError
    
    # called when a new array is generated
    def processArray(self, arr, attr):
        raise NotImplementedError
        
    # called when run offline
    def runOffline(self):
        import cv2
        cv2.namedWindow('result')

        # prepare input image
        fn = '/usr/share/doc/opencv-doc/examples/cpp/baboon.jpg'
        src = cv2.imread(fn)
        
        while True:
            # Change params
            print "Params: %s" % self
            param = raw_input("Param name to change (return to process image)? ")
            if param in self:
                typ = type(self[param])
                val = raw_input("Param value? ")
                try:
                    self[param] = typ(val)
                except Exception, e:
                    print "Cannot convert '%s' to %s" % (val, typ)
                    print e
            elif param:
                print "Invalid param name '%s'" % param
            
            # Run on image
            result = self.newArray(src)
            cv2.imshow('result', result)
            cv2.waitKey(500)

