/* Nasty nasty hack so that Python.h is happy. */
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#define _POSIX_C_SOURCE 200112L
#define _XOPEN_SOURCE 600
#include <Python.h>
#define PYTHON_USE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include <stdio.h>
#include <libgen.h>
#include <epicsTime.h>
#include "NDArray.h"
#include "adPythonPlugin.h"

const char *driverName = "adPythonPlugin";

adPythonPlugin::adPythonPlugin(const char *portNameArg, const char *filename,
                   const char *classname, int queueSize, int blockingCallbacks,
                   const char *NDArrayPort, int NDArrayAddr, int maxBuffers, size_t maxMemory,
                   int priority, int stackSize)
    : NDPluginDriver(portNameArg, queueSize, blockingCallbacks,
                       NDArrayPort, NDArrayAddr, 1, NUM_ADPYTHONPLUGIN_PARAMS, maxBuffers, maxMemory,
                       asynGenericPointerMask|asynFloat64ArrayMask,
                       asynGenericPointerMask|asynFloat64ArrayMask,
                       ASYN_MULTIDEVICE, 1, priority, stackSize)
{
    // Initialise some params
    this->pInstance = NULL;
    this->pProcessArray = NULL;
    this->pParamChanged = NULL;
    this->pParams = NULL;
    this->nextParam = 0;
    this->lastArray = NULL;
    this->pFileAttributes = new NDAttributeList;

    /* Create the epicsMutex for locking access to data structures from other threads */
    this->dictMutex = epicsMutexCreate();
    if (!this->dictMutex) {
        printf("%s::%s ERROR: epicsMutexCreate failure\n", driverName, __func__);
        return;
    }
    
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
    snprintf(buffer, sizeof(buffer), "PYTHONPATH=%s", DATADIR);
    //printf("%s\n", buffer);
    putenv(buffer);
    
    // Now we initialise python
    PyEval_InitThreads();
    Py_Initialize();
    
    // Be sure to save thread state otherwise other thread's PyGILState_Ensure()
    // calls will hang
    this->mainThreadState = PyEval_SaveThread();
    
    PyGILState_STATE state = PyGILState_Ensure();

    // Import the main dict
    this->pMain = PyImport_AddModule("__main__");
    if (this->pMain == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get __main__ module\n",
            driverName, __func__);
        PyGILState_Release(state);
        return;
    }
    this->pMainDict = PyModule_GetDict(this->pMain);
    if (this->pMainDict == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get __main__ dict\n",
            driverName, __func__);
        PyGILState_Release(state);
        return;
    }
    
    // Try and make an instance of this
    if (this->makePythonInstance()) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't make instance, user params not created\n",
            driverName, __func__);   
        PyGILState_Release(state); 
        return;
    }
    
    // Try and init numpy, this might fail if we didn't get a valid user program
    import_array();
    
    // Update param list from dict, also creating keys
    if (this->updateParams(1)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't update params, user params not created\n",
            driverName, __func__);    
        PyGILState_Release(state);
        return;    
    }    
    PyGILState_Release(state);
}

adPythonPlugin::~adPythonPlugin() {
    PyEval_RestoreThread(this->mainThreadState);
    Py_Finalize();
}

/** Import the user class from the pathname and make an instance of it */
asynStatus adPythonPlugin::makePythonInstance() {     
    char filename[BIGBUFFER], classname[BIGBUFFER], buffer[BIGBUFFER];
    printf("makePythonInstance\n");
    
    // Get the filename from param lib
    if (getStringParam(adPythonFilename, BIGBUFFER, filename)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get filename\n",
            driverName, __func__);
        return asynError;
    }
    
    // Get the classname from param lib
    if (getStringParam(adPythonClassname, BIGBUFFER, classname)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get classname\n",
            driverName, __func__);
        return asynError;
    }

    // Run python code for loading file from abs path
    snprintf(buffer, sizeof(buffer),
        "import imp, sys\n"
        "fname = '%s'\n"
        "f = open(fname)\n"        
        "try:\n"
        "   %s = imp.load_module('%s', f, fname, ('.py', 'U', 1)).%s\n"
        "finally:\n"
        "   f.close()\n", filename, classname, classname, classname);
    PyObject *pRet = PyRun_String(buffer, Py_file_input, this->pMainDict, this->pMainDict);
    if (pRet == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't import user class\n",
            driverName, __func__);
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
            driverName, __func__, classname);
        PyErr_PrintEx(0);
        return asynError;
    }
           
    // Create instance of this class
    Py_XDECREF(this->pInstance);
    this->pInstance = PyObject_CallObject(pCls, NULL);
    Py_DECREF(pCls);
    if (pInstance == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't make instance of class %s\n",
            driverName, __func__, classname);
        PyErr_PrintEx(0);
        return asynError;
    }

    // Get the processArray function ref
    Py_XDECREF(this->pProcessArray);
    this->pProcessArray = PyObject_GetAttrString(this->pInstance, "processArray");
    if (this->pProcessArray == NULL || !PyCallable_Check(this->pProcessArray)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get processArray ref\n",
            driverName, __func__);
        PyErr_PrintEx(0);
        return asynError;
    } 
    
    // Get the paramChanged function ref
    Py_XDECREF(this->pParamChanged);
    this->pParamChanged = PyObject_GetAttrString(this->pInstance, "paramChanged");
    if (this->pParamChanged == NULL || !PyCallable_Check(this->pParamChanged)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get paramChanged ref\n",
            driverName, __func__);
        PyErr_PrintEx(0);
        return asynError;
    } 
    
    // Get the param dict ref
    Py_XDECREF(this->pParams);
    this->pParams = PyObject_GetAttrString(this->pInstance, "_params");
    if (this->pParams == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get processArray ref\n",
            driverName, __func__);
        PyErr_PrintEx(0);
        return asynError;
    } 
    
    return asynSuccess;
}

/** Update instance param dict from param list */
asynStatus adPythonPlugin::updateDict() { 
    // Create param key list
    PyObject *pKeys = PyDict_Keys(this->pParams);
    if (pKeys == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get keys of _param dict\n",
            driverName, __func__);
        return asynError;
    }
    
    // Create a param of the correct type for each item
    for (Py_ssize_t i=0; i<PyList_Size(pKeys); i++) {
        int param;
        PyObject *key = PyList_GetItem(pKeys, i);
        PyObject *keyStr = PyObject_Str(key);
        char *paramStr = PyString_AsString(keyStr);
        if (findParam(paramStr, &param)) {
            asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: can't find param %s\n",
                driverName, __func__, paramStr);    
            continue;        
        }
        PyObject *pValue = PyDict_GetItem(this->pParams, key);
        if (PyFloat_Check(pValue)) {
            // get float param
            double value;
            getDoubleParam(param, &value);
            PyDict_SetItem(this->pParams, key, PyFloat_FromDouble(value));
        } else if (PyInt_Check(pValue)) {
            // get int param
            int value;
            getIntegerParam(param, &value);
            printf("Set %s %d\n", paramStr, value);
            PyDict_SetItem(this->pParams, key, PyInt_FromLong(value));
        } else if (PyString_Check(pValue)) {
            // get string param
            char value[BIGBUFFER];
            getStringParam(param, BIGBUFFER, value);
            PyDict_SetItem(this->pParams, key, PyString_FromString(value));
        } else {
            asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: param %s is not an int, float or string\n",
                driverName, __func__, paramStr);            
        }
    }
    Py_DECREF(pKeys);

    // call paramChanged method
    PyObject *pRet = PyObject_CallObject(this->pParamChanged, NULL);
    if (pRet == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: calling paramChanged failed\n",
            driverName, __func__);      
        PyErr_PrintEx(0);
        return asynError;
    }   
    Py_DECREF(pRet);
    
    return asynSuccess;
}

/** Update param list from instance param dict */
asynStatus adPythonPlugin::updateParams(int atinit) { 
    // Create param key list
    PyObject *pKeys = PyDict_Keys(this->pParams);
    if (pKeys == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
            "%s:%s: can't get keys of _param dict\n",
            driverName, __func__);
        return asynError;
    }
    
    // Create a param of the correct type for each item
    for (Py_ssize_t i=0; i<PyList_Size(pKeys); i++) {
        int param;
        PyObject *key = PyList_GetItem(pKeys, i);
        PyObject *keyStr = PyObject_Str(key);
        char *paramStr = PyString_AsString(keyStr);
        // If not at init, then find the param
        if (!atinit) {
            if (findParam(paramStr, &param)) {
                asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                    "%s:%s: can't find param %s\n",
                    driverName, __func__, paramStr);    
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
                driverName, __func__, paramStr);            
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
    // First call the base class method
    NDPluginDriver::processCallbacks(pArray);

    // Store the input array so we can reproduce it
    if (this->lastArray) this->lastArray->release();
    pArray->reserve();
    this->lastArray = pArray;
   
    // Process the NDArray
    this->processArray();

    // We have to modify our python dict to match our param list
    // so unlock and wait until any dictionary access has finished
    epicsMutexLock(this->dictMutex);
    if (this->dictModified) {
        // CA takes priority in dict writes
        this->dictModified = 0;
    } else {
        // Make sure we're allowed to use the python API
        PyGILState_STATE state = PyGILState_Ensure();
        // update param list, this will callParamCallbacks at the end
        this->updateParams(0);    
        // release GIL and dict Mutex
        PyGILState_Release(state);
    }
    epicsMutexUnlock(this->dictMutex);
    
    // Spit out the array
    if (this->pArrays[0]) {
        this->unlock();
        doCallbacksGenericPointer(this->pArrays[0], NDArrayData, 0);
        this->lock();    
    }
}

asynStatus adPythonPlugin::writeInt32(asynUser *pasynUser, epicsInt32 value) {
    asynStatus status;
    int param = pasynUser->reason;
    if (param == adPythonLoad || (this->nextParam && param > adPythonUserParams[0])) {
        // We have to modify our python dict to match our param list
        // so unlock and wait until any dictionary access has finished
        epicsMutexLock(this->dictMutex);
        this->dictModified = 1;
        // Make sure we're allowed to use the python API
        PyGILState_STATE state = PyGILState_Ensure();
        if (param == adPythonLoad) {
            // reload our python instance, this does callParamCallbacks for is
           setIntegerParam(param, 0);
           status = this->makePythonInstance();
        } else {
            // Now call the bast class to write the value to the param list
            status = NDPluginDriver::writeInt32(pasynUser, value);        
            // our param lib has changed, so update the dict and reprocess            
            this->updateDict();
        }
        // release GIL and dict Mutex
        PyGILState_Release(state);
        epicsMutexUnlock(this->dictMutex);        
    } else {
        status = NDPluginDriver::writeInt32(pasynUser, value);
    }
    return status;
}

asynStatus adPythonPlugin::writeFloat64(asynUser *pasynUser, epicsFloat64 value) {
    asynStatus status;
    int param = pasynUser->reason;
    if (this->nextParam && param > adPythonUserParams[0]) {
        // We have to modify our python dict to match our param list
        // so unlock and wait until any dictionary access has finished
        epicsMutexLock(this->dictMutex);
        this->dictModified = 1;
        // Make sure we're allowed to use the python API
        PyGILState_STATE state = PyGILState_Ensure();
        // Now call the bast class to write the value to the param list
        status = NDPluginDriver::writeFloat64(pasynUser, value);        
        // our param lib has changed, so update the dict and reprocess
        this->updateDict();
        // release GIL and dict Mutex
        PyGILState_Release(state);
        epicsMutexUnlock(this->dictMutex);                        
    } else {
        status = NDPluginDriver::writeFloat64(pasynUser, value);
    }
    return status;
}

asynStatus adPythonPlugin::writeOctet(asynUser *pasynUser, const char *value, size_t maxChars, size_t *nActual) {
    asynStatus status;
    int param = pasynUser->reason;
    if (this->nextParam && param > adPythonUserParams[0]) {
        // We have to modify our python dict to match our param list
        // so unlock and wait until any dictionary access has finished
        epicsMutexLock(this->dictMutex);
        this->dictModified = 1;
        // Make sure we're allowed to use the python API
        PyGILState_STATE state = PyGILState_Ensure();
        // Now call the bast class to write the value to the param list
        status = NDPluginDriver::writeOctet(pasynUser, value, maxChars, nActual);        
        // our param lib has changed, so update the dict and reprocess
        this->updateDict();
        // release GIL and dict Mutex
        PyGILState_Release(state);
        epicsMutexUnlock(this->dictMutex);       
    } else {
        status = NDPluginDriver::writeOctet(pasynUser, value, maxChars, nActual);
    }
    return status;
}

/* The obligatory lookup table of ad datatype to numpy datatype */
struct pix_lookup {
    int npy_fmt;
    NDDataType_t ad_fmt;
};

static const struct pix_lookup pix_lookup[] = {
   { NPY_INT8,     NDInt8 },     /**< Signed 8-bit integer */
   { NPY_UINT8,    NDUInt8 },    /**< Unsigned 8-bit integer */
   { NPY_INT16,    NDInt16 },    /**< Signed 16-bit integer */
   { NPY_UINT16,   NDUInt16 },   /**< Unsigned 16-bit integer */
   { NPY_INT32,    NDInt32 },    /**< Signed 32-bit integer */
   { NPY_UINT16,   NDUInt32 },   /**< Unsigned 32-bit integer */
   { NPY_FLOAT32,  NDFloat32 },  /**< 32-bit float */
   { NPY_FLOAT64,  NDFloat64 }   /**< 64-bit float */
};

/** Lookup a numpy pixel format from an NDDataType */
asynStatus adPythonPlugin::lookupNpyFormat(NDDataType_t ad_fmt, int *npy_fmt) {
    const int N = sizeof(pix_lookup) / sizeof(struct pix_lookup);
    for (int i = 0; i < N; i ++)
        if (ad_fmt == pix_lookup[i].ad_fmt) {
            *npy_fmt = pix_lookup[i].npy_fmt;
            return asynSuccess;
        }
    return asynError;
}

/** Lookup an NDDataType from a numpy pixel format */
asynStatus adPythonPlugin::lookupAdFormat(int npy_fmt, NDDataType_t *ad_fmt) {
    const int N = sizeof(pix_lookup) / sizeof(struct pix_lookup);
    for (int i = 0; i < N; i ++)
        if (npy_fmt == pix_lookup[i].npy_fmt) {
            *ad_fmt = pix_lookup[i].ad_fmt;
            return asynSuccess;
        }
    return asynError;
}

void adPythonPlugin::processArray() {      
    NDArrayInfo arrayInfo;
    
    // Return if no array to operate on
    if (this->lastArray == NULL) return;
    
    // Release the last produced array
    if (this->pArrays[0]) {
        this->pArrays[0]->release();    
        this->pArrays[0] = NULL;
    }
    
    // First store the time at the beginning of processing for profiling 
    epicsTimeStamp start, end;
    epicsTimeGetCurrent(&start);
    
    // Create a dimension description for numpy, note that we reverse dims
    npy_intp npy_dims[ND_ARRAY_MAX_DIMS];
    for (int i=0; i<this->lastArray->ndims; i++) {        
        npy_dims[i] = this->lastArray->dims[this->lastArray->ndims-i-1].size;
        //printf("npy_dims[%d] = %d\n", i, npy_dims[i]);
    }
    
    // Lookup the numpy format from the ad dataType of the array
    int npy_fmt;
    if (lookupNpyFormat(this->lastArray->dataType, &npy_fmt)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: can't lookup numpy format for dataType %d\n",
                driverName, __func__, this->lastArray->dataType);
        return;
    }    
        
     // Make sure we're allowed to use the python API
    PyGILState_STATE state = PyGILState_Ensure();
        
    // Wrap the existing data from the NDArray in a numpy array
    PyObject* pValue = PyArray_SimpleNewFromData(this->lastArray->ndims, npy_dims, npy_fmt, this->lastArray->pData);   
    if (pValue == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: Cannot make numpy array\n",
                driverName, __func__);      
        PyGILState_Release(state);                
        return;
    }     
    
    /* Construct an attribute list. We use a separate attribute list
    * from the one in pArray to avoid the need to copy the array. */
    /* First clear the list*/
    this->pFileAttributes->clear();
    
    /* Now get the current values of the attributes for this plugin */
    this->getAttributes(this->pFileAttributes);

    /* Now append the attributes from the array which are already up to date from
    * the driver and prior plugins */
    this->lastArray->pAttributeList->copy(this->pFileAttributes);    
           
    // Make a blank dict for the attributes
    PyObject* pDict = PyDict_New();
    if (pDict == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: Cannot make attribute dict\n",
                driverName, __func__);      
        Py_DECREF(pValue);
        PyGILState_Release(state);                
        return;
    } 
    
    // Fill it in
    NDAttribute *pAttr = this->pFileAttributes->next(NULL);
    while(pAttr != NULL) {    
        NDAttrDataType_t attrDataType;
        size_t attrDataSize;        
        PyObject *pObject = NULL;
        pAttr->getValueInfo(&attrDataType, &attrDataSize);
        void *value = calloc(1, attrDataSize);
        pAttr->getValue(attrDataType, value, attrDataSize);         
        /* If the attribute is a string, attrDataSize is the length of the string including the 0 terminator,
           otherwise it is the size in bytes of the specific data type */
        switch(attrDataType) {
            case(NDAttrInt8):
            case(NDAttrUInt8):
                pObject = PyInt_FromLong(*((short*) value));
                break;
            case(NDAttrInt16):
            case(NDAttrUInt16):
                pObject = PyInt_FromLong(*((int*) value));
                break;
            case(NDAttrInt32):
            case(NDAttrUInt32):
                pObject = PyInt_FromLong(*((long*) value)); 
                break;
            case(NDAttrFloat32):
                pObject = PyFloat_FromDouble(*((float*) value));
                break;
            case(NDAttrFloat64):
                pObject = PyFloat_FromDouble(*((double*) value));
                break;
            case(NDAttrString):
                pObject = PyString_FromString((char *) value);
                break;
            default:
                break;
        }
        free(value);
        if (pObject == NULL) {
            asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: attribute %s could not be put in attribute dict\n",
                driverName, __func__, pAttr->pName);            
        } else {            
            PyDict_SetItemString(pDict, pAttr->pName, pObject); 
            Py_DECREF(pObject); 
        }
        pAttr = this->pFileAttributes->next(pAttr);
    }    

    // Construct argument list, don't increment pValue so it is destroyed with
    // pArgs
    PyObject *pArgs = Py_BuildValue("(NO)", pValue, pDict);
    if (pArgs == NULL) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: Cannot build tuple for processArray()\n",
                driverName, __func__);      
        Py_DECREF(pValue);
        Py_DECREF(pDict);
        PyGILState_Release(state);
        return;
    } 
    
    // Unlock for long call
    this->unlock();
    
    // Make the function call
    pValue = PyObject_CallObject(this->pProcessArray, pArgs);
    
    Py_DECREF(pArgs);
    if (pValue == NULL) {
        Py_DECREF(pDict);
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: processArray() call failed again\n",
                driverName, __func__);      
        PyErr_PrintEx(0);
        PyGILState_Release(state);   
        this->lock();       
        return;
    }

    // Lock back up
    PyGILState_Release(state);
    this->lock();
    state = PyGILState_Ensure();
           
    // Check return type
    if (!PyObject_IsInstance(pValue, reinterpret_cast<PyObject*>(&PyArray_Type))) {
        // wasn't an array
        Py_DECREF(pValue);
        Py_DECREF(pDict);
        PyGILState_Release(state);
        return;
    }    
    
    // We must have an array, so find the dataType from it
    NDDataType_t ad_fmt;
    if (lookupAdFormat(PyArray_TYPE(pValue), &ad_fmt)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: can't lookup numpy format for dataType %d\n",
                driverName, __func__, this->lastArray->dataType);
        PyGILState_Release(state);
        return;
    }       
    
    // Create a dimension description from numpy array, note the inverse order
    size_t ad_dims[ND_ARRAY_MAX_DIMS];
    for (int i=0; i<PyArray_NDIM(pValue); i++) {
        ad_dims[i] = PyArray_DIMS(pValue)[PyArray_NDIM(pValue)-i-1];
    }
    
    /* Allocate the array */
    this->pArrays[0] = pNDArrayPool->alloc(PyArray_NDIM(pValue), ad_dims, ad_fmt, 0, NULL);
    if (this->pArrays[0] == NULL) {
        asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
                "%s:%s: error allocating buffer\n",
                driverName, __func__);
        PyGILState_Release(state);
        return;
    }

    // TODO: could avoid this memcpy if we could pass an existing
    // buffer to NDArray *AND* have it call a user free function
    this->pArrays[0]->getInfo(&arrayInfo);
    memcpy(this->pArrays[0]->pData, PyArray_DATA(pValue), arrayInfo.totalBytes);    
      
    // Fill in the pAttribute list from the dict
    Py_DECREF(pDict);
    
    // timestamp
    epicsTimeGetCurrent(&end);
    setDoubleParam(adPythonTime, epicsTimeDiffInSeconds(&end, &start)*1000);
    callParamCallbacks();     
    
    // done
    PyGILState_Release(state);    
}           


/** Configuration routine.  Called directly, or from the iocsh function in NDFileEpics */
static int adPythonPluginConfigure(const char *portNameArg, const char *filename,
                   const char *classname, int queueSize, int blockingCallbacks,
                   const char *NDArrayPort, int NDArrayAddr, int maxBuffers, size_t maxMemory,
                   int priority, int stackSize) {
    // Stack Size must be a minimum of 2MB
    if (stackSize < 2097152) stackSize = 2097152;
    new adPythonPlugin(portNameArg, filename,
                   classname, queueSize, blockingCallbacks,
                   NDArrayPort, NDArrayAddr, maxBuffers, maxMemory,
                   priority, stackSize);
    return(asynSuccess);
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

//extern "C" void adPythonPluginRegister(void);
static void adPythonPluginRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" {
epicsExportRegistrar(adPythonPluginRegister);
}
