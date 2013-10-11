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
    PyObject *pInstance = NULL;
    this->nParams = 0;
    this->initialised = 0;
    
    // Create the base class parameters (our python class may make some more)
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
    
    // Try and make an instance. Note that initialised=0 so we can make params
    this->makePythonInstance();
    
    // Mark the class as initialised so that we no longer can create parameters
    this->initialised = 1;
}

/** Wrap baseclass createParam so that we do the right thing when initialised */
virtual asynStatus adPythonPlugin::createParam(const char *name, asynParamType type, int *index) {
    asynStatus status;
    if (this->initialised) {
        // If initialised then just return already created param
        status = plugin->findParam(name, index);
    } else {
        // otherwise create a new param
        status = plugin->createParam(name, type, index);
        if (this->nParams < NUSERPARAMS) {
            this->adPythonUserParams[this->nParams++] = index;
        } else {
            status = AsynError;
        }
    }
    return status;
}

/** Import the user class from the pathname and make an instance of it */
asynStatus makePythonInstance() {     
    char filename[256], classname[256];
    getStringParam(adPythonFilename, &

    // Run python code for loading file from abs path
    char buffer[10000];
    snprintf(buffer, sizeof(buffer),
        "import imp, sys\n"
        "fname = '%s'\n"
        "try:\n"
        "   f = open(fname)\n"
        "   %s = imp.load_module('%s', f, fname, ('.py', 'U', 1)).%s\n"
        "finally:\n"
        "   f.close()\n", pth, cls, cls, cls);
    PyObject *pRet = PyRun_String(buffer, Py_file_input, this->pMainDict, this->pMainDict);

    if (pRet == NULL) {
        printf("Cannot import user module\n");
        PyErr_PrintEx(0);
        return NULL;
    }
    Py_DECREF(pRet);
        
    // Get the class name ref
    PyObject *pCls = PyMapping_GetItemString(this->pMainDict, cls);
    //Py_DECREF(this->pMainDict);
    if (pCls == NULL || !PyCallable_Check(pCls)) {
        printf("Cannot get class name ref %s\n", cls);
        PyErr_PrintEx(0);
        Py_XDECREF(pCls);
        return NULL;
    }
           
    PyObject *pArguments = Py_BuildValue("(O)", this->pCapsule);
    if (pArguments == NULL) {
        printf("Cannot make argument list\n");
        Py_DECREF(pCls);
        return NULL;
    }       
        
    PyObject *pInstance = PyObject_CallObject(pCls, pArguments);
    Py_DECREF(pCls);
    Py_DECREF(pArguments);
    if (pInstance == NULL) {
        printf("Cannot make instance of %s\n", cls);
        PyErr_PrintEx(0);
        return NULL;
    }
                
    // return this instance object
    return pInstance;
}

/** Callback function that is called by the NDArray driver with new NDArray data.
  * Does image statistics.
  * \param[in] pArray  The NDArray from the callback.
  */
void adPythonPlugin::processCallbacks(NDArray *pArray) {
    // First call the base class method
    NDPluginDriver::processCallbacks(pArray);
    
}

virtual asynStatus adPythonPlugin::writeInt32(asynUser *pasynUser, epicsInt32 value)
{
}

virtual asynStatus adPythonPlugin::writeFloat64(asynUser *pasynUser, epicsFloat64 value)
{
}



PyArrayObject * processArray(PyObject *pInstance, NDArray *pArray) {
    // Get the class name ref
    PyObject *pFunc = PyObject_GetAttrString(this->pInstance, "processArray");
    if (pFunc == NULL || !PyCallable_Check(pFunc)) {
        printf("Cannot get processArray ref\n");
        Py_XDECREF(pFunc);
        return NULL;
    } 
       
    // TODO: do this better
    npy_intp _sizes[CV_MAX_DIM+1];
    _sizes[0] = pArray->rows;
    _sizes[1] = pArray->cols;
    PyObject* pValue = PyArray_SimpleNewFromData(2, _sizes, NPY_UBYTE, pArray->pData);   
    if (pValue == NULL) {
        printf("Cannot make value\n");
        Py_DECREF(pFunc);
        return NULL;
    }     
    
    /* Make a blank dict */
    PyObject* pDict = PyDict_New();
    if (pDict == NULL) {
        printf("Cannot make dict\n");
        Py_DECREF(pFunc);
        Py_DECREF(pValue);
        return NULL;
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
        return NULL;
    } 
        
    // Make the function call
    pValue = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);    
    if (pValue == NULL) {
        PyErr_Print();
        Py_DECREF(pDict);
        fprintf(stderr,"Call failed\n");
        return NULL;
    }
    
    // Check return type
    if (!PyObject_IsInstance(pValue, reinterpret_cast<PyObject*>(&PyArray_Type))) {
        // wasn't an array
        Py_DECREF(pValue);
        Py_DECREF(pDict);
        return NULL;
    }    
    
    // Fill in the pAttribute list from the dict
    Py_DECREF(pDict);
    
    // return numpy array
    return (PyArrayObject*) pValue;
}           
            
int main(int argc, char *argv[]) {

    // Make an instance of morph class
    PyObject *pInstance = makeInstance(pth, (char *)"morph", cap);
    // Give it a new Array
    Mat img0;   
    img0 = imread("/usr/share/doc/opencv-doc/examples/cpp/baboon.jpg");
    PyArrayObject *pValue = processArray(pInstance, &img0);
    // Now parse the data output
    uchar * data = (uchar*)PyArray_DATA(pValue);
    printf("%i %i %i %i...\n", data[0], data[1], data[2], data[3]);
    // do it again
    sleep(10);
    pInstance = makeInstance(pth, (char *)"morph", cap);
    pValue = processArray(pInstance, &img0);
    // Now parse the data output
    data = (uchar*)PyArray_DATA(pValue);
    printf("%i %i %i %i...\n", data[0], data[1], data[2], data[3]);
    // Finish up
    Py_Finalize();
    return 0;
}


