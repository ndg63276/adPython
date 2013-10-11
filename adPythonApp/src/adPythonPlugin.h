#ifndef ADPYTHONPLUGIN_H
#define ADPYTHONPLUGIN_H

#include "Python.h"
#include "NDPluginDriver.h"

#define NUSERPARAMS 100

class adPythonPlugin : public NDPluginDriver {
public:
	adPythonPlugin(const char *portName, int queueSize, int blockingCallbacks,
				   const char *NDArrayPort, int NDArrayAddr, int maxBuffers, size_t maxMemory,
				   int priority, int stackSize, int numParams);
	~adPythonPlugin();
    /** This is called when the plugin gets a new array callback */
    virtual void processCallbacks(NDArray *pArray);
    /** This is when we get a new int value */
    virtual asynStatus writeInt32(asynUser *pasynUser, epicsInt32 value);
    /** This is when we get a new float value */
    virtual asynStatus writeFloat64(asynUser *pasynUser, epicsFloat64 value);
    /** CreateParam to be called from python that does error checking */
    virtual asynStatus createParam(const char *name, asynParamType type, int *index);
        
protected:
    /** These are the values of our parameters */
    #define FIRST_ADPYTHONPLUGIN_PARAM adPythonFilename
    int adPythonFilename;
    int adPythonClassname;
    int adPythonLoad;
    int adPythonTime;
    #define LAST_ADPYTHONPLUGIN_PARAM adPythonTime
    #define NUM_ADPYTHONPLUGIN_PARAMS (&LAST_ADPYTHONPLUGIN_PARAM - &FIRST_ADPYTHONPLUGIN_PARAM + 1 + NUSERPARAMS)
    int adPythonUserParams[NUSERPARAMS];

private:
    /** Load python file at pth, and create an instance of cls*/
    asynStatus makePythonInstance(const char* pth, const char* cls);
    PyObject *pCapsule;
    PyObject *pInstance;
    int initialised;
    int nextParam;
};

#endif
