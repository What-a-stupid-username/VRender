/* 
 * Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#pragma once

#include <optixu/optixpp_namespace.h>
#include <vector>

#include "sutilapi.h"


// Default catch block
#define SUTIL_CATCH( ctx ) catch( sutil::APIError& e ) {           \
    sutil::handleError( ctx, e.code, e.file.c_str(), e.line );     \
  }                                                                \
  catch( std::exception& e ) {                                     \
    sutil::reportErrorMessage( e.what() );                         \
    exit(1);                                                       \
  }

// Error check/report helper for users of the C API
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw sutil::APIError( code, __FILE__, __LINE__ );           \
  } while(0)

enum bufferPixelFormat
{
    BUFFER_PIXEL_FORMAT_DEFAULT, // The default depending on the buffer type
    BUFFER_PIXEL_FORMAT_RGB,     // The buffer is RGB or RGBA
    BUFFER_PIXEL_FORMAT_BGR,     // The buffer is BGR or BGRA
};

namespace sutil
{

// Exeption to be thrown by RT_CHECK_ERROR macro
struct APIError
{   
    APIError( RTresult c, const std::string& f, int l )
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

// Display error message
void SUTILAPI reportErrorMessage(
        const char* message);               // Error mssg to be displayed

// Queries provided RTcontext for last error message, displays it, then exits.
void SUTILAPI handleError(
        RTcontext context,                  // Context associated with the error
        RTresult code,                      // Code returned by OptiX API call
        const char* file,                   // Filename for error reporting
        int line);                          // File lineno for error reporting

// Query top level samples directory.
// The pointer returned may point to a static array.
SUTILAPI const char* samplesDir();

// Query directory containing PTX files for compiled sample CUDA files.
// The pointer returned may point to a static array.
SUTILAPI const char* samplesPTXDir();

// Create an output buffer with given specifications
optix::Buffer SUTILAPI createOutputBuffer(
        optix::Context context,             // optix context
        RTformat format,                    // Pixel format (must be ubyte4 for pbo)
        unsigned width,                     // Buffer width
        unsigned height,                    // Buffer height
        bool use_pbo);                      // Buffer type                    

// Create an input/output buffer with given specifications
optix::Buffer SUTILAPI createInputOutputBuffer(
    optix::Context context,             // optix context
    RTformat format,                    // Pixel format (must be ubyte4 for pbo)
    unsigned width,                     // Buffer width
    unsigned height,                    // Buffer height
    bool use_pbo);                      // Buffer type                    

// Resize a Buffer and its underlying GLBO if necessary
void SUTILAPI resizeBuffer(
        optix::Buffer buffer,               // Buffer to be modified
        unsigned width,                     // New buffer width
        unsigned height );                  // New buffer height

// Initialize GLUT.  Should be called before any GLUT display functions.
void SUTILAPI initGlut(
        int* argc,                          // Pointer to main argc param
        char** argv);                       // Pointer to main argv param

// Create GLUT window and display contents of the buffer.
void SUTILAPI displayBufferGlut(
        const char* window_title,           // Window title
        optix::Buffer buffer);              // Buffer to be displayed

// Create GLUT window and display contents of the buffer (C API version).
void SUTILAPI displayBufferGlut(
        const char* window_title,           // Window title
        RTbuffer buffer);                   // Buffer to be displayed

// Write the contents of the Buffer to a PPM image file
void SUTILAPI displayBufferPPM(
        const char* filename,                 // Image file to be created
        optix::Buffer buffer,                 // Buffer to be displayed
        bool disable_srgb_conversion = true); // Enables/disables srgb conversion before the image is saved. Disabled by default.            

// Write the contents of the Buffer to a PPM image file (C API version).
void SUTILAPI displayBufferPPM(
        const char* filename,                 // Image file to be created
        RTbuffer buffer,                      // Buffer to be displayed
        bool disable_srgb_conversion = true); // Enables/disables srgb conversion before the image is saved. Disabled by default.            

// Display contents of buffer, where the OpenGL/GLUT context is managed by caller.
void SUTILAPI displayBufferGL(
        optix::Buffer buffer,       // Buffer to be displayed
        bufferPixelFormat format = BUFFER_PIXEL_FORMAT_DEFAULT, // The pixel format of the buffer or 0 to use the default for the pixel type
        bool disable_srgb_conversion = false);

// Display frames per second, where the OpenGL/GLUT context
// is managed by the caller.
void SUTILAPI displayFps( unsigned total_frame_count );

// Display a short string starting at x,y.
void SUTILAPI displayText(const char* text, float x, float y);

// Create on OptiX TextureSampler for the given image file.  If the filename is
// empty or if loading the file fails, return 1x1 texture with default color.
optix::TextureSampler SUTILAPI loadTexture(
        optix::Context context,             // Context used for object creation 
        const std::string& filename,        // File to load
        optix::float3 default_color);       // Default color in case of file failure


// Creates a Buffer object for the given cubemap files.
optix::Buffer SUTILAPI loadCubeBuffer(
        optix::Context context,             // Context used for object creation
        const std::vector<std::string>& filenames ); // Files to be loaded 


// Calculate appropriate U,V,W for pinhole_camera shader.
void SUTILAPI calculateCameraVariables(
        optix::float3 eye,                  // Camera eye position
        optix::float3 lookat,               // Point in scene camera looks at
        optix::float3 up,                   // Up direction
        float  fov,                         // Horizontal or vertical field of view (assumed horizontal, see boolean below)
        float  aspect_ratio,                // Pixel aspect ratio (width/height)
        optix::float3& U,                   // [out] U coord for camera program
        optix::float3& V,                   // [out] V coord for camera program
        optix::float3& W,                   // [out] W coord for camera program
		bool fov_is_vertical = false );

// Blocking sleep call
void SUTILAPI sleep(
        int seconds );                      // Number of seconds to sleep

// Parse the string of the form <width>x<height> and return numeric values.
void SUTILAPI parseDimensions(
        const char* arg,                    // String of form <width>x<height>
        int& width,                         // [out] width
        int& height );                      // [in]  height

// Get current time in seconds for benchmarking/timing purposes.
double SUTILAPI currentTime();

// Get PTX, either pre-compiled with NVCC or JIT compiled by NVRTC.
SUTILAPI const char* getPtxString(
        const char* sample,                 // Name of the sample, used to locate the input file. NULL = only search the common /cuda dir
        const char* filename,               // Cuda C input file name
        const char** log = NULL );          // (Optional) pointer to compiler log string. If *log == NULL there is no output. Only valid until the next getPtxString call

// Ensures that width and height have the minimum size to prevent launch errors.
void SUTILAPI ensureMinimumSize(
    int& width,                             // Will be assigned the minimum suitable width if too small.
    int& height);                           // Will be assigned the minimum suitable height if too small.

// Ensures that width and height have the minimum size to prevent launch errors.
void SUTILAPI ensureMinimumSize(
    unsigned& width,                        // Will be assigned the minimum suitable width if too small.
    unsigned& height);                      // Will be assigned the minimum suitable height if too small.

} // end namespace sutil

