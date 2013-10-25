# our base class requires numpy, so make sure it's on the path here
# this step is only needed if numpy is an egg installed multi-version
try:
    from pkg_resources import require
    require("numpy")
except:
    pass

# define a helper function that imports a python filename and returns an 
# instance of classname which is contained in it
def makePythonInstance(filename, classname):
    import imp
    try:
        f = open(filename)
        mod = imp.load_module('%s', f, filename, ('.py', 'U', 1))
        f.close()
        inst = getattr(mod, classname)()
        inst.paramChanged()
        return inst
    except Exception, e:
        print "Creating %s:%s threw exception %s" % (filename, classname, e)

class AdPythonPlugin(object):   
    # Will be our param dict
    _params = None
    
    # init our param dict
    def __init__(self, params={}):
        self._params = dict(params)

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
        fn = '/home/tmc43/bwtest-sm.jpg'
        src = cv2.imread(fn, 0)
        
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
            result = self.processArray(src, {})
            cv2.imshow('result', result)
            cv2.waitKey(500)

