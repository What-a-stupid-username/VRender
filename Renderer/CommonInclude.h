#pragma once

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#    include <GL/freeglut.h>
#  else
#    include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "../cuda/DataBridge.h"
#include <sutil.h>
#include <Arcball.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <functional>

#include "VDebug.h"

using namespace optix;
using namespace std;

//this define only exist in optix6.0+
#ifdef rtMarkedCallableProgramId

#define OPTIX_6

#endif


//Use this to shut down the RTCore.
//Currently, RTCore can't support change the count of the Intersection Program in runtime...Shut down it to fix a bug in develop mode.
#define FORCE_NOT_USE_RTX