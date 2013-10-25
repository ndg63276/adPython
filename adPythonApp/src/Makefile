TOP=../..

include $(TOP)/configure/CONFIG

# -------------------------------
# Build an Diamond Support Module
# -------------------------------

USR_CXXFLAGS += -Werror
USR_CXXFLAGS += -Wall -Wextra
USR_CXXFLAGS += -Wno-unused-parameter
USR_CXXFLAGS += -Wno-missing-field-initializers
# Can't do this because of base/include/epicsTypes.h
#USR_CXXFLAGS += -Wundef
# Can't do this because of base/include/locationException.h
#USR_CXXFLAGS += -Wshadow
USR_CXXFLAGS += -Wcast-align
USR_CXXFLAGS += -Wwrite-strings
USR_CXXFLAGS += -Wredundant-decls
USR_CXXFLAGS += -Wmissing-declarations
USR_CXXFLAGS += -Wcast-qual



LIBRARY_IOC += adPython

# xxxRecord.dbd will be installed into <top>/dbd
DBD += adPythonPlugin.dbd

# The following are compiled and added to the support library
adPython_SRCS += adPythonPlugin.cpp

# link against python
ifneq ($(PYTHON),)
	USR_INCLUDES += -I$(PYTHON)/include/python2.7
	python2.7_DIR = $(PYTHON)/lib
	adPython_LIBS += python2.7
	USR_CXXFLAGS += -DDATADIR=\"$(shell cd $(TOP)/data; pwd)\"
	#pyDrv_SYS_LIBS += util
else
	pyDrv_SYS_LIBS += python
endif

DATA += $(patsubst ../%, %, $(wildcard ../*.py))

# We need to link against the EPICS Base libraries
#xxx_LIBS += $(EPICS_BASE_IOC_LIBS)

include $(TOP)/configure/RULES
