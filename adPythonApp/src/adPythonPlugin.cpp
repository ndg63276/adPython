#define PYTHON_USE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
//#define MODULESTR "cv2"
#include "numpy/ndarrayobject.h"
//#include "opencv2/opencv.hpp"
#include <stdio.h>
//using namespace cv;
//using namespace std;
#include <libgen.h>
#include <epicsTime.h>

const char *driverName = "adPythonPlugin"

adPythonPlugin::adPythonPlugin(const char *portName, const char *filename,
                   const char *classname, int queueSize, int blockingCallbacks,
				   const char *NDArrayPort, int NDArrayAddr, int maxBuffers, size_t maxMemory,
				   int priority, int stackSize);
	: NDPluginDriver(portName, queueSize, blockingCallbacks,
					   NDArrayPort, NDArrayAddr, 1, NUM_ADPYTHONPLUGIN_PARAMS, maxBuffers, maxMemory,
					   asynGenericPointerMask|asynFloat64ArrayMask,
					   asynGenericPointerMask|asynFloat64ArrayMask,
					   ASYN_MULTIDEVICE, 1, priority, stackSize)
{
    // Initialise some params
    static const char *functionName = "adPythonPlugin";
    Py_ssize_t i;
    this->pInstance = NULL;
    this->pProcessArray = NULL;
    this->pParamChanged = NULL;
    this->pParams = NULL;
    this->nextParam = 0;
    this->lastArray = NULL;
    
    // Create the base class parameters (our python class may make some more)
    setStringParam(NDPluginDriverPluginType, driverName);
	createParam("ADPYTHON_FILENAME",   asynParamOctet,   &adPythonFilename);
	setStringParam(adPythonFilename, filename);
    createParam("ADPYTHON_CLASSNAME",  asynParamOctet,   &adPythonClassname);
    setStringParam(adPythonClassname, classname);
    createParam("ADPYTHON_LOAD",       asynParamInt32,   &adPythonLoad);
    createParam("ADPYTHON_TIME",       asynParamFloat64, &adPythonTime);    

    // First we tell python where to find adPythonPlugin.py
    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "PYTHONPATH=%s", dirname(strdup(__FILE__)));
    putenv(buffer);
    
    // Now we initialise python, numpy, and our python lib
    Py_Initialize();
    _import_array();
    initadPythonParamLib();

    // Create a capsule containing this
    this->pCapsule = PyCapsule_New(this, "adPythonPlugin", NULL);
    if (this->pCapsule == NULL) {
    	asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't create PyCapsule\n",
			driverName, functionName);
        return;
    };
    
    // Import the main dict
    this->pMain = PyImport_AddModule("__main__");
    if (main == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get __main__ module\n",
			driverName, functionName);
        return;
    }
    this->pMainDict = PyModule_GetDict(main);
    if (this->pMainDict == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get __main__ dict\n",
			driverName, functionName);
        return;
    }
    
    // Try and make an instance of this
    if (this->makePythonInstance()) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't make instance, user params not created\n",
			driverName, functionName);    
	    return;
    }
    
    // Update param list from dict, also creating keys
    if (this->updateParams(1)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't update params, user params not created\n",
			driverName, functionName);    
	    return;    
	}
}

/** Import the user class from the pathname and make an instance of it */
asynStatus adPythonPlugin::makePythonInstance() {     
    static const char *functionName = "makePythonInstance";
    char filename[BIGBUFFER], classname[BIGBUFFER], buffer[BIGBUFFER];
    
    // Get the filename from param lib
    if (getStringParam(adPythonFilename, BIGBUFFER, filename)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
		    "%s:%s: can't get filename\n",
		    driverName, functionName);
		return AsynError;
    }
    
    // Get the classname from param lib
    if (getStringParam(adPythonFilename, BIGBUFFER, classname)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
		    "%s:%s: can't get classname\n",
		    driverName, functionName);
		return AsynError;
    }

    // Run python code for loading file from abs path
    snprintf(buffer, sizeof(buffer),
        "import imp, sys\n"
        "fname = '%s'\n"
        "try:\n"
        "   f = open(fname)\n"
        "   %s = imp.load_module('%s', f, fname, ('.py', 'U', 1)).%s\n"
        "finally:\n"
        "   f.close()\n", filename, classname, classname, classname);
    PyObject *pRet = PyRun_String(buffer, Py_file_input, this->pMainDict, this->pMainDict);
    if (pRet == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't import user class\n",
			driverName, functionName);
        PyErr_PrintEx(0);
        return asynError;
    }
    Py_DECREF(pRet);
        
    // Get the class name ref
    PyObject *pCls = PyMapping_GetItemString(this->pMainDict, classname);
    if (pCls == NULL || !PyCallable_Check(pCls)) {
        Py_XDECREF(pCls);    
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get class name ref %s\n",
			driverName, functionName, classname);
        PyErr_PrintEx(0);
        return asynError;
    }
           
    // Create instance of this class
    Py_XDECREF(this->pInstance);
    this->pInstance = PyObject_CallObject(pCls);
    Py_DECREF(pCls);
    if (pInstance == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't make instance of class\n",
			driverName, functionName, classname);
        PyErr_PrintEx(0);
        return asynError;
    }

    // Get the processArray function ref
    Py_XDECREF(this->pProcessArray);
    this->pProcessArray = PyObject_GetAttrString(this->pInstance, "processArray");
    if (this->pProcessArray == NULL || !PyCallable_Check(this->pProcessArray)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get processArray ref\n",
			driverName, functionName, classname);
        PyErr_PrintEx(0);
        return asynError;
    } 
    
    // Get the paramChanged function ref
    Py_XDECREF(this->pParamChanged);
    this->pParamChanged = PyObject_GetAttrString(this->pInstance, "paramChanged");
    if (this->pParamChanged == NULL || !PyCallable_Check(this->pParamChanged)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get paramChanged ref\n",
			driverName, functionName, classname);
        PyErr_PrintEx(0);
        return asynError;
    } 
    
    // Get the param dict ref
    Py_XDECREF(this->pParam);
    this->pParam = PyObject_GetAttrString(this->pInstance, "pParam");
    if (this->pParam == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get processArray ref\n",
			driverName, functionName, classname);
        PyErr_PrintEx(0);
        return asynError;
    } 
    
    return asynSuccess;
}

/** Update instance param dict from param list */
asynStatus adPythonPlugin::updateDict() { 
    static const char *functionName = "updateDict";
    // Create param key list
    PyObject *pKeys = PyDict_Keys(this->pParams);
    if (pKeys == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get keys of _param dict\n",
			driverName, functionName);
        return asynError;
    }
    
    // Create a param of the correct type for each item
    for (i=0; i<PyList_Size(keys); i++) {
        int param;
        PyObject *key = PyList_GetItem(keys, i);
        PyObject *keyStr = PyObject_Str(key);
        char *paramStr = PyString_AsString(keyStr);
        if (findParam(paramStr, &param) {
            asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
	    		"%s:%s: can't find param %s\n",
	    		driverName, functionName, paramStr);    
	        continue;        
        }
        PyObject *value = PyDict_SetItem(this->pParams, key);
        if (PyFloat_Check(value)) {
            // get float param
            double value;
            getDoubleParam(param, &value);
            PyDict_SetItem(this->pParams, key, PyFloat_FromDouble(value));
        } else if (PyInt_Check(value)) {
            // get int param
            int value;
            getIntegerParam(param, &value);
            PyDict_SetItem(this->pParams, key, PyInt_FromLong(value));
        } else if (PyString_Check(value)) {
            // get string param
            char value[BIGBUFFER];
            getStringParam(param, BIGBUFFER, &value);
            PyDict_SetItem(this->pParams, key, PyString_FromString(value));
        } else {
            asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
		    	"%s:%s: param %s is not an int, float or string\n",
		    	driverName, functionName, param);            
        }
    }
    Py_DECREF(pKeys);

    // call paramChanged method
    PyObject *pRet = PyObject_CallObject(this->pParamChanged, NULL);
    if (pRet == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
	    	"%s:%s: calling paramChanged failed\n",
	    	driverName, functionName);      
	    PyErr_PrintEx(0);
	    return asynError;
	}   
    Py_DECREF(pRet);
    
    return asynSuccess;
}

/** Update param list from instance param dict */
asynStatus adPythonPlugin::updateParams(int atinit) { 
    static const char *functionName = "updateParams";
    // Create param key list
    PyObject *pKeys = PyDict_Keys(this->pParams);
    if (pKeys == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
			"%s:%s: can't get keys of _param dict\n",
			driverName, functionName);
        return asynError;
    }
    
    // Create a param of the correct type for each item
    for (i=0; i<PyList_Size(keys); i++) {
        int param;
        PyObject *key = PyList_GetItem(keys, i);
        PyObject *keyStr = PyObject_Str(key);
        char *paramStr = PyString_AsString(keyStr);
        // If not at init, then find the param
        if (!atinit) {
            if (findParam(paramStr, &param) {
                asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
	        		"%s:%s: can't find param %s\n",
	        		driverName, functionName, paramStr);    
	         continue;        
            }
        }
        PyObject *value = PyDict_GetItem(this->pParams, key);
        if (PyFloat_Check(value)) {
            if (atinit) {
                createParam(paramStr, asynParamFloat64, &adPythonUserParams[this->nextParam]);
                param = adPythonUserParams[this->nextParam++];
            }
            // set float param
            setDoubleParam(param, PyFloat_AsDouble(value));
        } else if (PyInt_Check(value)) {
            if (atinit) {
                createParam(paramStr, asynParamInt32, &adPythonUserParams[this->nextParam]);
                param = adPythonUserParams[this->nextParam++];
            }
            // set int param
            setIntegerParam(param, PyInt_AsLong(value));
        } else if (PyString_Check(value)) {            
            if (atinit) {
                createParam(paramStr, asynParamOctet, &adPythonUserParams[this->nextParam]);
                param = adPythonUserParams[this->nextParam++];
            }
            // set string param
            setStringParam(param, PyString_AsString(value));
        } else {
            asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
		    	"%s:%s: param %s is not an int, float or string\n",
		    	driverName, functionName, param);            
        }
    }
    Py_DECREF(pKeys);
    callParamCallbacks();
    return asynSuccess;
}


/** Callback function that is called by the NDArray driver with new NDArray data.
  * Does image statistics.
  * \param[in] pArray  The NDArray from the callback.
  */
void adPythonPlugin::processCallbacks(NDArray *pArray) {
    static const char *functionName = "processCallbacks";
    // First call the base class method
    NDPluginDriver::processCallbacks(pArray);
    
    // Store the input array so we can reproduce it
    if (this->lastArray) this->lastArray->release();
    
    // now call the processArray method which will call the python
    this->processArray();
}

virtual asynStatus adPythonPlugin::writeInt32(asynUser *pasynUser, epicsInt32 value) {
    static const char *functionName = "writeInt32";
    // First call the base class method
    asynStatus status = NDPluginDriver::writeInt32(pasynUser, value)
    int param = pasynUser->reason;
    if (param == adPythonLoad) {
        setIntegerParam(param, 0);
        status = this->makePythonInstance();
    }
    if (param == adPythonLoad || (this->nextParam && param > adPythonUserParams[0])) {
        // our param lib has changed, so update the dict and reprocess
        this->updateDict();
        this->processArray();
    }
    return status
}

virtual asynStatus adPythonPlugin::writeFloat64(asynUser *pasynUser, epicsFloat64 value)
    static const char *functionName = "writeFloat64";
    // First call the base class method
    asynStatus status = NDPluginDriver::writeFloat64(pasynUser, value)
    int param = pasynUser->reason;
    if (this->nextParam && param > adPythonUserParams[0]) {
        // our param lib has changed, so update the dict and reprocess
        this->updateDict();
        this->processArray();
    }
    return status
}

virtual asynStatus adPythonPlugin::writeOctet(asynUser *pasynUser, const char *value, size_t maxChars, size_t *nActual)
    static const char *functionName = "writeOctet";
    // First call the base class method
    asynStatus status = NDPluginDriver::writeOctet(pasynUser, value, maxChars, nActual)
    int param = pasynUser->reason;
    if (this->nextParam && param > adPythonUserParams[0]) {
        // our param lib has changed, so update the dict and reprocess
        this->updateDict();
        this->processArray();
    }
    return status
}

void adPythonPlugin::processArray() {      
    static const char *functionName = "processArray";
    // First store the time at the beginning of processing for profiling 
    epicsTimeStamp start, end;
    epicsTimeGetCurrent(&start);
    
    // Create a numpy wrapper to the input array
    npy_intp _sizes[3];
        
    // TODO: do this better
    npy_intp _sizes[CV_MAX_DIM+1];
    _sizes[0] = 300;
    _sizes[1] = 400;
    PyObject* pValue = PyArray_SimpleNewFromData(2, _sizes, NPY_UBYTE, pArray->pData);   
    if (pValue == NULL) {
        printf("Cannot make value\n");
        Py_DECREF(pFunc);
        return;
    }     
    
    /* Make a blank dict */
    PyObject* pDict = PyDict_New();
    if (pDict == NULL) {
        printf("Cannot make dict\n");
        Py_DECREF(pFunc);
        Py_DECREF(pValue);
        return;
    } 
    
    // Fill it in
    PyDict_SetItemString(pDict, "foo", "bar");

    // Construct argument list, don't increment pValue so it is destroyed with
    // pArgs
    PyObject *pArgs = Py_BuildValue("(NO)", pValue, pDict);
    if (pArgs == NULL) {
        printf("Cannot make tuple\n");
        Py_DECREF(pFunc);
        Py_DECREF(pValue);
        Py_DECREF(pDict);
        return;
    } 
        
    // Unlock for potentially long call
    this->unlock();
        
    // Make the function call
    pValue = PyObject_CallObject(pFunc, pArgs);
    this->lock();
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);    
    if (pValue == NULL) {
        PyErr_Print();
        Py_DECREF(pDict);
        fprintf(stderr,"Call failed\n");
        return;
    }
    
    // Check return type
    if (!PyObject_IsInstance(pValue, reinterpret_cast<PyObject*>(&PyArray_Type))) {
        // wasn't an array
        Py_DECREF(pValue);
        Py_DECREF(pDict);
        return;
    }    
    
    // Fill in the pAttribute list from the dict
    Py_DECREF(pDict);
    
    // Now parse the data output
    uchar * data = (uchar*)PyArray_DATA(pValue);    
    
    // timestamp
    epicsTimeGetCurrent(&end);
    setDoubleParam(adPythonTime, epicsTimeDiffInSeconds(&end, &start));
    
    // update param list, this will callParamCallbacks at the end
    this->updateParams(0);
}           

/* EPICS iocsh shell commands */
static const iocshArg initArg0 = { "portName",iocshArgString};
static const iocshArg initArg1 = { "filename",iocshArgString};
static const iocshArg initArg2 = { "classname",iocshArgString};
static const iocshArg initArg3 = { "frame queue size",iocshArgInt};
static const iocshArg initArg4 = { "blocking callbacks",iocshArgInt};
static const iocshArg initArg5 = { "NDArrayPort",iocshArgString};
static const iocshArg initArg6 = { "NDArrayAddr",iocshArgInt};
static const iocshArg initArg7 = { "maxBuffers",iocshArgInt};
static const iocshArg initArg8 = { "maxMemory",iocshArgInt};
static const iocshArg initArg9 = { "priority",iocshArgInt};
static const iocshArg initArg10 = { "stackSize",iocshArgInt};
static const iocshArg * const initArgs[] = {&initArg0,
                                            &initArg1,
                                            &initArg2,
                                            &initArg3,
                                            &initArg4,
                                            &initArg5,
                                            &initArg6,
                                            &initArg7,
                                            &initArg8,
                                            &initArg9,
                                            &initArg10};
static const iocshFuncDef initFuncDef = {"adPythonPluginConfigure",11,initArgs};
static void initCallFunc(const iocshArgBuf *args)
{
	adPythonPluginConfigure(args[0].sval, args[1].sval, args[2].sval, 
	                   args[3].ival, args[4].ival,
                       args[5].sval, args[6].ival, args[7].ival,
                       args[8].ival, args[9].ival, args[10].ival);
}

extern "C" void adPythonPluginRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" {
epicsExportRegistrar(adPythonPluginRegister);
}
