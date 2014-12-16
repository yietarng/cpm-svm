
HEADERS += \
    svm.h \
    linear_algebra.h \
    data.h \
    solve_qp.h

SOURCES += \
    svm.cpp \
    main.cpp \
    data.cpp \
    solve_qp.cpp

# cgal
# LIBS += -lCGAL -lgmp

CONFIG -= qt


# mosek library
INCLUDEPATH += /home/sergei/DevTools/mosek/7/tools/platform/linux32x86/h
LIBS += -lpthread
LIBS += -L/home/sergei/DevTools/mosek/7/tools/platform/linux32x86/bin -lmosek -liomp5


#debug {
#    DEFINES += BMRM_INFO
#}
