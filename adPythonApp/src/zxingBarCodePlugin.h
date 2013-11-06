#ifndef ZXIINGBARCODEPLUGIN_H
#define ZXIINGBARCODEPLUGIN_H

#include "NDPluginDriver.h"
#include <iocsh.h>
#include <epicsExport.h>


class zxingBarCodePlugin : public NDPluginDriver {
public:
	zxingBarCodePlugin(const char *portName, int queueSize, int blockingCallbacks,
				   const char *NDArrayPort, int NDArrayAddr, int maxBuffers, size_t maxMemory,
				   int priority, int stackSize);
	~zxingBarCodePlugin() {}
    /** This is called when the plugin gets a new array callback */
    virtual void processCallbacks(NDArray *pArray);
    /** This is when we get a new int value */
    virtual asynStatus writeInt32(asynUser *pasynUser, epicsInt32 value);
    /** This is when we get a new float value */
    //virtual asynStatus writeFloat64(asynUser *pasynUser, epicsFloat64 value);
    /** This is when we get a new string value */
    //virtual asynStatus writeOctet(asynUser *pasynUser, const char *value, size_t maxChars, size_t *nActual);
        
protected:
    /** These are the values of our parameters */
    #define FIRST_ZXBC_PARAM zxbc_busy
    int zxbc_busy;
    int zxbc_data;
    int zxbc_type;
    int zxbc_count;
    int zxbc_location_x1;
    int zxbc_location_y1;
    int zxbc_location_x2;
    int zxbc_location_y2;
    int zxbc_location_x3;
    int zxbc_location_y3;
    int zxbc_location_x4;
    int zxbc_location_y4;
    #define LAST_ZXBC_PARAM zxbc_location_y4
    #define NUM_ZXBC_PARAMS (&LAST_ZXBC_PARAM - &FIRST_ZXBC_PARAM + 1)

private:

};

#endif
