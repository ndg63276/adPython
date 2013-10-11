TOP=../..

include $(TOP)/configure/CONFIG

# -------------------------------
# Build an Diamond Support Module
# -------------------------------

LIBRARY_IOC += adPython

# xxxRecord.dbd will be installed into <top>/dbd
DBD += adPythonPlugin.dbd

# The following are compiled and added to the support library
adPython_SRCS += adPythonPlugin.cpp

# link against python
ifneq ($(PYTHON),)
	USR_INCLUDES += -I$(PYTHON)/include
	python2.7_DIR = $(PYTHON)/lib
	adPython_LIBS += python2.7
	#pyDrv_SYS_LIBS += util
else
	pyDrv_SYS_LIBS += python
endif

# We need to link against the EPICS Base libraries
#xxx_LIBS += $(EPICS_BASE_IOC_LIBS)

include $(TOP)/configure/RULES
