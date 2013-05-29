#pragma once

#include <ml.h>
#include <assert.h>

#define ES_Error(msg) CV_Error(0, msg)
#define ES_Assert(expression) assert(expression)