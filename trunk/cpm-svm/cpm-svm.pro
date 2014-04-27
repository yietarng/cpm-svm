
HEADERS += \
    svm.h \
    linear_algebra.h \
    data.h

SOURCES += \
    svm.cpp \
    main.cpp \
    data.cpp

LIBS += -lCGAL -lgmp

CONFIG -= qt
