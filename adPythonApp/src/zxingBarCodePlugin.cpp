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
    createParam("zxbc_busy",        asynParamInt32,   &zxbc_busy);
    createParam("zxbc_data",        asynParamOctet,   &zxbc_data);
    createParam("zxbc_type",        asynParamOctet,   &zxbc_type);
    createParam("zxbc_count",       asynParamInt32,   &zxbc_count);
    createParam("zxbc_location_x1",  asynParamFloat64, &zxbc_location_x1);
    createParam("zxbc_location_y1",  asynParamFloat64, &zxbc_location_y1);
    createParam("zxbc_location_x2",  asynParamFloat64, &zxbc_location_x2);
    createParam("zxbc_location_y2",  asynParamFloat64, &zxbc_location_y2);
    createParam("zxbc_location_x3",  asynParamFloat64, &zxbc_location_x3);
    createParam("zxbc_location_y3",  asynParamFloat64, &zxbc_location_y3);
    createParam("zxbc_location_x4",  asynParamFloat64, &zxbc_location_x4);
    createParam("zxbc_location_y4",  asynParamFloat64, &zxbc_location_y4);

    setIntegerParam(zxbc_busy, 0);
    setStringParam(zxbc_data,   "");
    setStringParam(zxbc_type,  "");
    callParamCallbacks();
}

void zxingBarCodePlugin::processCallbacks(NDArray *pArray)
{
    // First call the base class method
    NDPluginDriver::processCallbacks(pArray);

    // Check datatype. Only support 8bit grey
    if (!(pArray->dataType == NDInt8 or pArray->dataType == NDUInt8)) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_ERROR, "zxing only support 8bpp greyscale images. Aborting scan.\n");
        return;
    }

    NDArrayInfo_t arr_info;
    pArray->getInfo(&arr_info);

    // Convert the buffer to something that the library understands.
    Ref<LuminanceSource> source (new BufferBitmapSource(arr_info.xSize, arr_info.ySize, (char*)pArray->pData));
    // Turn it into a binary image.
    Ref<Binarizer> binarizer (new GlobalHistogramBinarizer(source));
    Ref<BinaryBitmap> image(new BinaryBitmap(binarizer));
    
    NDArray *pArrOut = NULL;
    try {
        Ref<BitMatrix> matrix(image->getBlackMatrix());

        // Extract the thresholded/histogrammed binary image that zxing is analysing to find the barcode
        size_t dims[2];
        dims[0] = (size_t)image->getWidth();
        dims[1] = (size_t)image->getHeight();
        pArrOut = this->pNDArrayPool->alloc(2, dims, NDUInt8, 0, NULL);
        for (int x = 0; x < matrix->getWidth(); x++) {
        	for (int y = 0; y < matrix->getHeight(); y++) {
        		*((unsigned char*)(pArrOut->pData)+(y*matrix->getWidth() + x)) = matrix->get(x,y) ? 0 : 255;
        	}
        }
    } catch (zxing::Exception& e) {
        asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW, "Found no BlackMatrix. Zxing says: \"%s\"\n", e.what());
    }

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
        asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW, "Found no symbols. Zxing says: \"%s\"\n", e.what());
        count = 0;
    }

    if (count > 0) {
        // Output the DataMatrix result.
        asynPrint(this->pasynUserSelf, ASYN_TRACE_FLOW, "SUCCESS: found %d hits!\n" \
        			"         Type: \"%s\"\n" \
        			"         Data: \"%s\"\n",
        			count, BarcodeFormat::barcodeFormatNames[dm_result->getBarcodeFormat()],
        			dm_result->getText()->getText().c_str());
        int busy = 0;
        getIntegerParam(zxbc_busy, &busy);
        if (busy == 1) {
        	// Only update the results if the user has requested a scan.
        	// Otherwise we might accidentally overwrite a previous result.
            setStringParam(zxbc_data, dm_result->getText()->getText().c_str());
            setStringParam(zxbc_type, BarcodeFormat::barcodeFormatNames[dm_result->getBarcodeFormat()]);
            setIntegerParam(zxbc_count, count);

            setDoubleParam(zxbc_location_x1, dm_result->getResultPoints()[0]->getX() );
            setDoubleParam(zxbc_location_y1, dm_result->getResultPoints()[0]->getY() );
            setDoubleParam(zxbc_location_x2, dm_result->getResultPoints()[1]->getX() );
            setDoubleParam(zxbc_location_y2, dm_result->getResultPoints()[1]->getY() );
            setDoubleParam(zxbc_location_x3, dm_result->getResultPoints()[2]->getX() );
            setDoubleParam(zxbc_location_y3, dm_result->getResultPoints()[2]->getY() );
            setDoubleParam(zxbc_location_x4, dm_result->getResultPoints()[3]->getX() );
            setDoubleParam(zxbc_location_y4, dm_result->getResultPoints()[3]->getY() );
            setIntegerParam(zxbc_busy, 0); // Done now
            callParamCallbacks();
        }
    } else {
        // Nothing found
    }

    // Output the binary bitmap image as a result
    if (pArrOut) {
        this->unlock();
        doCallbacksGenericPointer(pArrOut, NDArrayData, 0);
        this->lock();
        pArrOut->release();
    }

    callParamCallbacks();
}


asynStatus zxingBarCodePlugin::writeInt32(asynUser *pasynUser, epicsInt32 value)
{
	int status = asynSuccess;
	int param = pasynUser->reason;
	if (param == zxbc_busy) {
		int previous = 0;
		getIntegerParam(zxbc_busy, &previous);
		if (value==1 && value != previous) {
			// User has requested a new scan to start so clear the previous results
			asynPrint(pasynUserSelf, ASYN_TRACE_FLOW, "User started scan. Clearing results.\n");
			setStringParam(zxbc_data, "");
			setStringParam(zxbc_type, "");
			setIntegerParam(zxbc_count, 0);
			setDoubleParam(zxbc_location_x1, 0.0);
			setDoubleParam(zxbc_location_y1, 0.0);
			setDoubleParam(zxbc_location_x2, 0.0);
			setDoubleParam(zxbc_location_y2, 0.0);
			setDoubleParam(zxbc_location_x3, 0.0);
			setDoubleParam(zxbc_location_y3, 0.0);
			setDoubleParam(zxbc_location_x4, 0.0);
			setDoubleParam(zxbc_location_y4, 0.0);
		}
		setIntegerParam(zxbc_busy, value);
	} else {
		status |= NDPluginDriver::writeInt32(pasynUser, value);
	}
	callParamCallbacks();
	return (asynStatus) status;
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
