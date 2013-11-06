/*
 * zxingBarCodePlugin.cpp
 *
 *  Created on: 1 Nov 2013
 *      Author: Ulrik Kofoed Pedersen
 */
#include <stdio.h>
#include <epicsTime.h>
#include <NDArray.h>
#include <NDPluginDriver.h>

// ZXing includes
#include <zxing/qrcode/QRCodeReader.h>
#include <zxing/datamatrix/DataMatrixReader.h>
#include <zxing/Exception.h>
#include <zxing/common/GlobalHistogramBinarizer.h>
#include <zxing/DecodeHints.h>

#include "zxingBarCodePlugin.h"
#include "BufferBitmapSource.h"

using namespace std;
using namespace zxing;
using namespace zxing::qrcode;
using namespace zxing::datamatrix;
using namespace qrviddec;

const char *zxDriverName = "zxingBarCodePlugin";

zxingBarCodePlugin::zxingBarCodePlugin(const char *portNameArg, 
        int queueSize, int blockingCallbacks,
        const char *NDArrayPort, int NDArrayAddr, int maxBuffers,
        size_t maxMemory, int priority, int stackSize)
: NDPluginDriver(portNameArg, queueSize, blockingCallbacks,
                   NDArrayPort, NDArrayAddr, 1, NUM_ZXBC_PARAMS,
                   maxBuffers, maxMemory,
                   asynGenericPointerMask|asynFloat64ArrayMask,
                   asynGenericPointerMask|asynFloat64ArrayMask,
                   ASYN_MULTIDEVICE, 1, priority, stackSize)
{
    // Create the base class parameters (our python class may make some more)
    setStringParam(NDPluginDriverPluginType, zxDriverName);
    createParam("zxbc_data",        asynParamOctet,   &zxbc_data);
    createParam("zxbc_type",        asynParamOctet,   &zxbc_type);
    createParam("zxbc_count",       asynParamInt32,   &zxbc_count);
    createParam("zxbc_location_x",  asynParamFloat64, &zxbc_location_x);    
    createParam("zxbc_location_y",  asynParamFloat64, &zxbc_location_y);    
    createParam("zxbc_size_x",      asynParamFloat64, &zxbc_size_x);    
    createParam("zxbc_size_y",      asynParamFloat64, &zxbc_size_y);    

    setStringParam(zxbc_data,   "");
    setStringParam(zxbc_type,  "");
    callParamCallbacks();
}

void zxingBarCodePlugin::processCallbacks(NDArray *pArray) {
    // First call the base class method
    NDPluginDriver::processCallbacks(pArray);

    // Convert the buffer to something that the library understands.
    Ref<LuminanceSource> source (new BufferBitmapSource(pArray->dims[1].size, pArray->dims[0].size, (char*)pArray->pData));
    // Turn it into a binary image.
    Ref<Binarizer> binarizer (new GlobalHistogramBinarizer(source));
    Ref<BinaryBitmap> image(new BinaryBitmap(binarizer));
    Ref<BitMatrix> matrix(image->getBlackMatrix());

    // Tell the decoder to try as hard as possible.
    DecodeHints hints(DecodeHints::DATA_MATRIX_HINT);
    hints.setTryHarder(true);

    // Perform the DataMatrix decoding.
    DataMatrixReader dm_reader;
    Ref<Result> dm_result;
    int count = 0;
    try {
        dm_result=(dm_reader.decode(image, hints));
        count = dm_result->count();
    } catch (zxing::Exception& e)
    {
        // Did not find a symbol in the image
        asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW, "%s\n", e.what());
    }

    if (count > 0) {
        // Output the DataMatrix result.
        asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW, "SUCCESS: found %d hits!\n" \
        "         Type: %s\n" \
        "         Data: \"%s\"\n",
        count, BarcodeFormat::barcodeFormatNames[dm_result->getBarcodeFormat()], 
        BarcodeFormat::barcodeFormatNames[dm_result->getBarcodeFormat()]);

        setStringParam(zxbc_data, dm_result->getText()->getText().c_str());
        setStringParam(zxbc_type, BarcodeFormat::barcodeFormatNames[dm_result->getBarcodeFormat()]);
        setIntegerParam(zxbc_count, count);
        
        setDoubleParam(zxbc_location_x, dm_result->getResultPoints()[0]->getX() );
        setDoubleParam(zxbc_location_y, dm_result->getResultPoints()[0]->getY() );
        //setDoubleParam(zxbc_size_x, );
        //setDoubleParam(zxbc_size_y, );
        callParamCallbacks();       
    } else {
        // Nothing found
    }
}

/** Configuration routine.  Called directly, or from the iocsh function in NDFileEpics */
static int zxingBarCodeConfigure(const char *portNameArg, int queueSize, int blockingCallbacks,
                   const char *NDArrayPort, int NDArrayAddr, int maxBuffers, size_t maxMemory,
                   int priority, int stackSize) {
    // Stack Size must be a minimum of 2MB
    if (stackSize < 2097152) stackSize = 2097152;
    new zxingBarCodePlugin(portNameArg, queueSize, blockingCallbacks,
                   NDArrayPort, NDArrayAddr, maxBuffers, maxMemory,
                   priority, stackSize);
    return(asynSuccess);
}

/* EPICS iocsh shell commands */
static const iocshArg initArg0 = { "portName",iocshArgString};
static const iocshArg initArg1 = { "frame queue size",iocshArgInt};
static const iocshArg initArg2 = { "blocking callbacks",iocshArgInt};
static const iocshArg initArg3 = { "NDArrayPort",iocshArgString};
static const iocshArg initArg4 = { "NDArrayAddr",iocshArgInt};
static const iocshArg initArg5 = { "maxBuffers",iocshArgInt};
static const iocshArg initArg6 = { "maxMemory",iocshArgInt};
static const iocshArg initArg7 = { "priority",iocshArgInt};
static const iocshArg initArg8 = { "stackSize",iocshArgInt};
static const iocshArg * const initArgs[] = {&initArg0,
                                            &initArg1,
                                            &initArg2,
                                            &initArg3,
                                            &initArg4,
                                            &initArg5,
                                            &initArg6,
                                            &initArg7,
                                            &initArg8};
static const iocshFuncDef initFuncDef = {"zxingBarCodeConfigure",9,initArgs};
static void initCallFunc(const iocshArgBuf *args)
{
	zxingBarCodeConfigure(args[0].sval,
                       args[1].ival, args[2].ival,
                       args[3].sval, args[4].ival, args[5].ival,
                       args[6].ival, args[7].ival, args[8].ival);
}

static void zxingBarCodePluginRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" {
epicsExportRegistrar(zxingBarCodePluginRegister);
}
