# our base class requires numpy, so make sure it's on the path here
# this step is only needed if numpy is an egg installed multi-version
try:
    from pkg_resources import require
    require("numpy")
except:
    pass

import os, logging
#logging.basicConfig(format='%(asctime)s %(levelname)8s %(name)20s:  %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s %(levelname)8s %(name)20s  %(filename)s:%(lineno)d %(funcName)s():  %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s %(levelname)8s %(filename)20s:%(lineno)d %(funcName)16s():  %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)8s %(name)20s  %(filename)s:%(lineno)d:  %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

# define a helper function that imports a python filename and returns an 
# instance of classname which is contained in it
def makePythonInstance(filename, classname):
    import imp
    try:
        f = open(filename)
        pymodule, ext = os.path.splitext(os.path.basename(filename))
        mod = imp.load_module(pymodule, f, filename, (ext, 'U', 1))
        f.close()
        inst = getattr(mod, classname)()
        inst.log = logging.getLogger(".".join([pymodule, classname]))
        inst.paramChanged()
        return inst
    except Exception, e:
        log.exception( "Creating %s:%s threw exception", filename, classname )


class AdPythonPlugin(object):   
    # Will be our param dict
    _params = None
    
    # init our param dict
    def __init__(self, params={}):
        # Parent class logger
        self._log = logging.getLogger(".".join([__name__, self.__class__.__name__]))
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
            self._log.info( "Params: %s", self )
            param = raw_input("Param name to change (return to process image)? ")
            if param in self:
                typ = type(self[param])
                val = raw_input("Param value? ")
                try:
                    self[param] = typ(val)
                except Exception, e:
                    # The exception information is automatically added 
                    # to a Logger.exception() msg
                    self._log.exception("Cannot convert '%s' to %s", val, typ) 
            elif param:
                self._log.warning("Invalid param name '%s'", param )
            
            # Run on image
            result = self.processArray(src, {})
            cv2.imshow('result', result)
            cv2.waitKey(500)

