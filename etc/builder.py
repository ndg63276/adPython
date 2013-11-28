from iocbuilder import Device, AutoSubstitution
from iocbuilder.arginfo import *

from iocbuilder.modules.areaDetector import AreaDetector, _NDPluginBase

class AdPython(Device):
    '''Library dependencies for adPython'''
    Dependencies = (AreaDetector,)
    # Device attributes
    LibFileList = ['adPython']
    DbdFileList = ['adPythonPlugin']
    AutoInstantiate = True

class _adPythonBase(AutoSubstitution):
    '''This plugin Works out the area and tip of a sample'''
    TemplateFile = "adPythonPlugin.template"

class adPythonPlugin(_NDPluginBase):
    """This plugin creates an adPython object"""
    _SpecificTemplate = _adPythonBase
    Dependencies = (AdPython,)

    def __init__(self, classname, BUFFERS = 50, MEMORY = 0, **args):
        # Init the superclass (_NDPluginBase)
        self.__super.__init__(**args)
        # Init the python classname specific class
        class _tmp(AutoSubstitution):
            ModuleName = adPythonPlugin.ModuleName
            TrueName = "_adPython%s" % classname
            TemplateFile = "adPython%s.template" % classname
        _tmp(**filter_dict(args, _tmp.ArgInfo.Names()))
        # Store the args
        self.filename = "$(ADPYTHON)/adPythonApp/scripts/adPython%s.py" % classname
        self.__dict__.update(locals())

    def Initialise(self):
        print '# %(Configure)s(portName, filename, classname, queueSize, '\
            'blockingCallbacks, NDArrayPort, NDArrayAddr, maxBuffers, ' \
            'maxMemory)' % self.__dict__
        print '%(Configure)s("%(PORT)s", "%(filename)s", "%(classname)s", %(QUEUE)d, ' \
            '%(BLOCK)d, "%(NDARRAY_PORT)s", %(NDARRAY_ADDR)s, %(BUFFERS)d, ' \
            '%(MEMORY)d)' % self.__dict__

    # __init__ arguments
    ArgInfo = _NDPluginBase.ArgInfo + makeArgInfo(__init__,
        classname = Choice('Predefined python class to use', ["Morph", "Focus", "Template", "BarCode"]),
        BUFFERS = Simple('Maximum number of NDArray buffers to be created for '
            'plugin callbacks', int),
        MEMORY = Simple('Max memory to allocate, should be maxw*maxh*nbuffer '
            'for driver and all attached plugins', int))


