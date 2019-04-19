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


// Note: wglew.h has to be included before sutil.h on Windows
#if defined(__APPLE__)
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/glut.h>
#endif

#include <sutil/sutil.h>
#include <sutil/HDRLoader.h>
#include <sutil/PPMLoader.h>
#include <sampleConfig.h>

#include <optixu/optixu_math_namespace.h>

#include <nvrtc.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <sstream>
#include <map>
#include <memory>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN 1
#    endif
#    include<windows.h>
#    include<mmsystem.h>
#else // Apple and Linux both use this
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif


using namespace optix;


namespace
{

// Global variables for GLUT display functions
RTcontext g_context                  = 0;
RTbuffer  g_image_buffer             = 0;
bufferPixelFormat g_image_buffer_format = BUFFER_PIXEL_FORMAT_DEFAULT;
bool      g_glut_initialized         = false;
bool      g_disable_srgb_conversion  = false;


// Converts the buffer format to gl format
GLenum glFormatFromBufferFormat(bufferPixelFormat pixel_format, RTformat buffer_format)
{
    if (buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
    {
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_BGRA;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGRA;
        default:
            throw Exception("Unknown buffer pixel format");
        }
    }
    else if (buffer_format == RT_FORMAT_FLOAT4)
    {
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGBA;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGRA;
        default:
            throw Exception("Unknown buffer pixel format");
        }
    }
    else if (buffer_format == RT_FORMAT_FLOAT3)
        switch (pixel_format)
        {
        case BUFFER_PIXEL_FORMAT_DEFAULT:
            return GL_RGB;
        case BUFFER_PIXEL_FORMAT_RGB:
            return GL_RGB;
        case BUFFER_PIXEL_FORMAT_BGR:
            return GL_BGR;
        default:
            throw Exception("Unknown buffer pixel format");
        }
    else if (buffer_format == RT_FORMAT_FLOAT)
        return GL_LUMINANCE;
    else
        throw Exception("Unknown buffer format");
}


void keyPressed(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27: // esc
        case 'q':
            rtContextDestroy( g_context );
            exit(EXIT_SUCCESS);
    }
}


void checkBuffer( RTbuffer buffer )
{
    // Check to see if the buffer is two dimensional
    unsigned int dimensionality;
    RT_CHECK_ERROR( rtBufferGetDimensionality(buffer, &dimensionality) );
    if (2 != dimensionality)
        throw Exception( "Attempting to display non-2D buffer" );

    // Check to see if the buffer is of type float{1,3,4} or uchar4
    RTformat format;
    RT_CHECK_ERROR( rtBufferGetFormat(buffer, &format) );
    if( RT_FORMAT_FLOAT  != format &&
            RT_FORMAT_FLOAT4 != format &&
            RT_FORMAT_FLOAT3 != format &&
            RT_FORMAT_UNSIGNED_BYTE4 != format )
        throw Exception( "Attempting to display buffer with format not float, float3, float4, or uchar4");
}


void displayBuffer()
{
    optix::Buffer buffer = Buffer::take( g_image_buffer );

    // Query buffer information
    RTsize buffer_width_rts, buffer_height_rts;
    buffer->getSize( buffer_width_rts, buffer_height_rts );
    uint32_t width  = static_cast<int>(buffer_width_rts);
    uint32_t height = static_cast<int>(buffer_height_rts);
    RTformat buffer_format = buffer->getFormat();

    GLboolean use_SRGB = GL_FALSE;
    if( !g_disable_srgb_conversion && (buffer_format == RT_FORMAT_FLOAT4 || buffer_format == RT_FORMAT_FLOAT3) )
    {
        glGetBooleanv( GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &use_SRGB );
        if( use_SRGB )
            glEnable(GL_FRAMEBUFFER_SRGB_EXT);
    }

    static unsigned int gl_tex_id = 0;
    if( !gl_tex_id )
    {
        glGenTextures( 1, &gl_tex_id );
        glBindTexture( GL_TEXTURE_2D, gl_tex_id );

        // Change these to GL_LINEAR for super- or sub-sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    glBindTexture( GL_TEXTURE_2D, gl_tex_id );

    // send PBO or host-mapped image data to texture
    const unsigned pboId = buffer->getGLBOId();
    GLvoid* imageData = 0;
    if( pboId )
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboId );
    else
        imageData = buffer->map( 0, RT_BUFFER_MAP_READ );

    RTsize elmt_size = buffer->getElementSize();
    if      ( elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if ( elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if ( elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    GLenum pixel_format = glFormatFromBufferFormat(g_image_buffer_format, buffer_format);

    if( buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, pixel_format, GL_UNSIGNED_BYTE, imageData);
    else if(buffer_format == RT_FORMAT_FLOAT4)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else if(buffer_format == RT_FORMAT_FLOAT3)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else if(buffer_format == RT_FORMAT_FLOAT)
        glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, pixel_format, GL_FLOAT, imageData );
    else
        throw Exception( "Unknown buffer format" );

    if( pboId )
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    else
        buffer->unmap();

    // 1:1 texel to pixel mapping with glOrtho(0, 1, 0, 1, -1, 1) setup:
    // The quad coordinates go from lower left corner of the lower left pixel
    // to the upper right corner of the upper right pixel.
    // Same for the texel coordinates.

    glEnable(GL_TEXTURE_2D);

    glBegin(GL_QUADS);
    glTexCoord2f( 0.0f, 0.0f );
    glVertex2f( 0.0f, 0.0f );

    glTexCoord2f( 1.0f, 0.0f );
    glVertex2f( 1.0f, 0.0f);

    glTexCoord2f( 1.0f, 1.0f );
    glVertex2f( 1.0f, 1.0f );

    glTexCoord2f(0.0f, 1.0f );
    glVertex2f( 0.0f, 1.0f );
    glEnd();

    glDisable(GL_TEXTURE_2D);

    if ( use_SRGB )
        glDisable(GL_FRAMEBUFFER_SRGB_EXT);
}


void displayBufferSwap()
{
    displayBuffer();
    glutSwapBuffers();
}


void SavePPM(const unsigned char *Pix, const char *fname, int wid, int hgt, int chan)
{
    if( Pix==NULL || wid < 1 || hgt < 1 )
        throw Exception( "Image is ill-formed. Not saving" );

    if( chan != 1 && chan != 3 && chan != 4 )
        throw Exception( "Attempting to save image with channel count != 1, 3, or 4.");

    std::ofstream OutFile(fname, std::ios::out | std::ios::binary);
    if(!OutFile.is_open())
        throw Exception( "Could not open file for SavePPM" );

    bool is_float = false;
    OutFile << 'P';
    OutFile << ((chan==1 ? (is_float?'Z':'5') : (chan==3 ? (is_float?'7':'6') : '8'))) << std::endl;
    OutFile << wid << " " << hgt << std::endl << 255 << std::endl;

    OutFile.write(reinterpret_cast<char*>(const_cast<unsigned char*>( Pix )), wid * hgt * chan * (is_float ? 4 : 1));

    OutFile.close();
}


bool dirExists( const char* path )
{
#if defined(_WIN32)
    DWORD attrib = GetFileAttributes( path );
    return (attrib != INVALID_FILE_ATTRIBUTES) && (attrib & FILE_ATTRIBUTE_DIRECTORY);
#else
    DIR* dir = opendir( path );
    if( dir == NULL )
        return false;

    closedir(dir);
    return true;
#endif
}

} // end anonymous namespace


void sutil::reportErrorMessage( const char* message )
{
    std::cerr << "OptiX Error: '" << message << "'\n";
#if defined(_WIN32) && defined(RELEASE_PUBLIC)
    {
        char s[2048];
        sprintf( s, "OptiX Error: %s", message );
        MessageBox( 0, s, "OptiX Error", MB_OK|MB_ICONWARNING|MB_SYSTEMMODAL );
    }
#endif
}


void sutil::handleError( RTcontext context, RTresult code, const char* file,
        int line)
{
    const char* message;
    char s[2048];
    rtContextGetErrorString(context, code, &message);
    sprintf(s, "%s\n(%s:%d)", message, file, line);
    reportErrorMessage( s );
	system("PAUSE");
}


const char* sutil::samplesDir()
{
    static char s[512];

    // Allow for overrides.
    const char* dir = getenv( "OPTIX_SAMPLES_SDK_DIR" );
    if (dir) {
        strcpy(s, dir);
        return s;
    }

    // Return hardcoded path if it exists.
    if( dirExists( SAMPLES_DIR ) )
        return SAMPLES_DIR;

    // Last resort.
    return ".";
}


const char* sutil::samplesPTXDir()
{
    static char s[512];

    // Allow for overrides.
    const char* dir = getenv( "OPTIX_SAMPLES_SDK_PTX_DIR" );
    if (dir) {
        strcpy(s, dir);
        return s;
    }

    // Return hardcoded path if it exists.
    if( dirExists(SAMPLES_PTX_DIR) )
        return SAMPLES_PTX_DIR;

    // Last resort.
    return ".";
}

optix::Buffer createBufferImpl(
        optix::Context context,
        RTformat format,
        unsigned width,
        unsigned height,
        bool use_pbo,
        RTbuffertype buffer_type)
{
    optix::Buffer buffer;
    if( use_pbo )
    {
        // First allocate the memory for the GL buffer, then attach it to OptiX.

        // Assume ubyte4 or float4 for now
        unsigned int elmt_size = format == RT_FORMAT_UNSIGNED_BYTE4 ?  4 : 16;

        GLuint vbo = 0;
        glGenBuffers( 1, &vbo );
        glBindBuffer( GL_ARRAY_BUFFER, vbo );
        glBufferData( GL_ARRAY_BUFFER, elmt_size * width * height, 0, GL_STREAM_DRAW);
        glBindBuffer( GL_ARRAY_BUFFER, 0 );

        buffer = context->createBufferFromGLBO(buffer_type, vbo);
        buffer->setFormat( format );
        buffer->setSize( width, height );
    }
    else
    {
        buffer = context->createBuffer( buffer_type, format, width, height );
    }

    return buffer;
}

optix::Buffer sutil::createOutputBuffer(
    optix::Context context,
    RTformat format,
    unsigned width,
    unsigned height,
    bool use_pbo)
{
    return createBufferImpl(context, format, width, height, use_pbo, RT_BUFFER_OUTPUT);
}


optix::Buffer SUTILAPI sutil::createInputOutputBuffer(optix::Context context, 
    RTformat format, 
    unsigned width, 
    unsigned height, 
    bool use_pbo)
{
    return createBufferImpl(context, format, width, height, use_pbo, RT_BUFFER_INPUT_OUTPUT);
}

void sutil::resizeBuffer( optix::Buffer buffer, unsigned width, unsigned height )
{
    buffer->setSize( width, height );

    // Check if we have a GL interop display buffer
    const unsigned pboId = buffer->getGLBOId();
    if( pboId )
    {
        buffer->unregisterGLBuffer();
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboId );
        glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        buffer->registerGLBuffer();
    }
}


void sutil::initGlut( int* argc, char** argv)
{
    // Initialize GLUT
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DOUBLE );
    glutInitWindowSize( 100, 100 );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( argv[0] );
    g_glut_initialized = true;
}


void sutil::displayBufferGlut( const char* window_title, Buffer buffer )
{
    displayBufferGlut(window_title, buffer->get() );
}


void sutil::displayBufferGlut( const char* window_title, RTbuffer buffer )
{
    if( !g_glut_initialized )
        throw Exception( "displayGlutWindow called before initGlut.");

    checkBuffer(buffer);
    g_image_buffer = buffer;

    RTsize buffer_width, buffer_height;
    RT_CHECK_ERROR( rtBufferGetSize2D(buffer, &buffer_width, &buffer_height ) );

    GLsizei width  = static_cast<int>( buffer_width );
    GLsizei height = static_cast<int>( buffer_height );
    glutSetWindowTitle(window_title);
    glutReshapeWindow( width, height );

    // Init state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutKeyboardFunc(keyPressed);
    glutDisplayFunc(displayBufferSwap);
    glutSwapBuffers();

    glutMainLoop();
}


void sutil::displayBufferPPM( const char* filename, Buffer buffer, bool disable_srgb_conversion)
{
    displayBufferPPM( filename, buffer->get(), disable_srgb_conversion );
}

void write_bmpheader(unsigned char *bitmap, int offset, int bytes, int value) {
	int i;
	for (i = 0; i < bytes; i++)
		bitmap[offset + i] = (value >> (i << 3)) & 0xFF;
}

int SaveBMP(unsigned char *image, int imageWidth, int imageHeight, const char *filename)
{
	unsigned char header[54] = {
		0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
		54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 32, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0
	};

	long file_size = (long)imageWidth * (long)imageHeight * 4 + 54;
	header[2] = (unsigned char)(file_size & 0x000000ff);
	header[3] = (file_size >> 8) & 0x000000ff;
	header[4] = (file_size >> 16) & 0x000000ff;
	header[5] = (file_size >> 24) & 0x000000ff;

	long width = imageWidth;
	header[18] = width & 0x000000ff;
	header[19] = (width >> 8) & 0x000000ff;
	header[20] = (width >> 16) & 0x000000ff;
	header[21] = (width >> 24) & 0x000000ff;

	long height = imageHeight;
	header[22] = height & 0x000000ff;
	header[23] = (height >> 8) & 0x000000ff;
	header[24] = (height >> 16) & 0x000000ff;
	header[25] = (height >> 24) & 0x000000ff;

	char fname_bmp[128];
	sprintf(fname_bmp, "%s.bmp", filename);

	FILE *fp;
	if (!(fp = fopen(fname_bmp, "wb")))
		return -1;

	fwrite(header, sizeof(unsigned char), 54, fp);
	fwrite(image, sizeof(unsigned char), (size_t)(long)imageWidth * imageHeight * 4, fp);

	fclose(fp);
	return 0;
}




void sutil::displayBufferBMP(const char* filename, optix::Buffer buffer, bool disable_srgb_conversion) {
	auto rtbuffer = buffer->get();
	GLsizei width, height;
	RTsize buffer_width, buffer_height;

	GLvoid* imageData;
	RT_CHECK_ERROR(rtBufferMap(rtbuffer, &imageData));

	RT_CHECK_ERROR(rtBufferGetSize2D(rtbuffer, &buffer_width, &buffer_height));
	width = static_cast<GLsizei>(buffer_width);
	height = static_cast<GLsizei>(buffer_height);

	std::vector<unsigned char> pix(width * height * 4);

	RTformat buffer_format;
	RT_CHECK_ERROR(rtBufferGetFormat(rtbuffer, &buffer_format));

	const float gamma_inv = 1.0f / 2.2f;

	switch (buffer_format) {
	case RT_FORMAT_UNSIGNED_BYTE4:
		// Data is BGRA and upside down, so we need to swizzle to RGB
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + (3 * width*(height - 1 - j));
			unsigned char *src = ((unsigned char*)imageData) + (4 * width*j);
			for (int i = 0; i < width; i++) {
				*dst++ = *(src + 2);
				*dst++ = *(src + 1);
				*dst++ = *(src + 0);
				src += 4;
			}
		}
		break;

	case RT_FORMAT_FLOAT:
		// This buffer is upside down
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + width * (height - 1 - j);
			float* src = ((float*)imageData) + (3 * width*j);
			for (int i = 0; i < width; i++) {
				int P;
				if (disable_srgb_conversion)
					P = static_cast<int>((*src++) * 255.0f);
				else
					P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
				unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;

				// write the pixel to all 3 channels
				*dst++ = static_cast<unsigned char>(Clamped);
				*dst++ = static_cast<unsigned char>(Clamped);
				*dst++ = static_cast<unsigned char>(Clamped);
			}
		}
		break;

	case RT_FORMAT_FLOAT3:
		// This buffer is upside down
		for (int j = height - 1; j >= 0; --j) {
			unsigned char *dst = &pix[0] + (3 * width*(height - 1 - j));
			float* src = ((float*)imageData) + (3 * width*j);
			for (int i = 0; i < width; i++) {
				for (int elem = 0; elem < 3; ++elem) {
					int P;
					if (disable_srgb_conversion)
						P = static_cast<int>((*src++) * 255.0f);
					else
						P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
					unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
					*dst++ = static_cast<unsigned char>(Clamped);
				}
			}
		}
		break;

	case RT_FORMAT_FLOAT4:
		// This buffer is upside down
		for (int j = 0; j <= height - 1; ++j) {
			unsigned char *dst = &pix[0] + (4 * width*j);
			float* src = ((float*)imageData) + (4 * width*j);
			for (int i = 0; i < width; i++) {
				for (int elem = 0; elem < 4; ++elem) {
					int P;
					if (disable_srgb_conversion)
						P = static_cast<int>((*src++) * 255.0f);
					else
						P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
					unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
					*dst++ = static_cast<unsigned char>(Clamped);
				}

				// skip alpha
				//src++;
			}
		}
		break;

	default:
		fprintf(stderr, "Unrecognized buffer data type or format.\n");
		exit(2);
		break;
	}

	SaveBMP(&pix[0], height, width, filename);
	//SavePPM(&pix[0], filename, width, height, 3);


	// Now unmap the buffer
	RT_CHECK_ERROR(rtBufferUnmap(rtbuffer));
}

void sutil::displayBufferPPM( const char* filename, RTbuffer buffer, bool disable_srgb_conversion)
{
    GLsizei width, height;
    RTsize buffer_width, buffer_height;

    GLvoid* imageData;
    RT_CHECK_ERROR( rtBufferMap( buffer, &imageData) );

    RT_CHECK_ERROR( rtBufferGetSize2D(buffer, &buffer_width, &buffer_height) );
    width  = static_cast<GLsizei>(buffer_width);
    height = static_cast<GLsizei>(buffer_height);

    std::vector<unsigned char> pix(width * height * 3);

    RTformat buffer_format;
    RT_CHECK_ERROR( rtBufferGetFormat(buffer, &buffer_format) );

    const float gamma_inv = 1.0f / 2.2f;

    switch(buffer_format) {
        case RT_FORMAT_UNSIGNED_BYTE4:
            // Data is BGRA and upside down, so we need to swizzle to RGB
            for(int j = height-1; j >= 0; --j) {
                unsigned char *dst = &pix[0] + (3*width*(height-1-j));
                unsigned char *src = ((unsigned char*)imageData) + (4*width*j);
                for(int i = 0; i < width; i++) {
                    *dst++ = *(src + 2);
                    *dst++ = *(src + 1);
                    *dst++ = *(src + 0);
                    src += 4;
                }
            }
            break;

        case RT_FORMAT_FLOAT:
            // This buffer is upside down
            for(int j = height-1; j >= 0; --j) {
                unsigned char *dst = &pix[0] + width*(height-1-j);
                float* src = ((float*)imageData) + (3*width*j);
                for(int i = 0; i < width; i++) {
                    int P;
                    if (disable_srgb_conversion)
                        P = static_cast<int>((*src++) * 255.0f);
                    else
                        P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
                    unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;

                    // write the pixel to all 3 channels
                    *dst++ = static_cast<unsigned char>(Clamped);
                    *dst++ = static_cast<unsigned char>(Clamped);
                    *dst++ = static_cast<unsigned char>(Clamped);
                }
            }
            break;

        case RT_FORMAT_FLOAT3:
            // This buffer is upside down
            for(int j = height-1; j >= 0; --j) {
                unsigned char *dst = &pix[0] + (3*width*(height-1-j));
                float* src = ((float*)imageData) + (3*width*j);
                for(int i = 0; i < width; i++) {
                    for(int elem = 0; elem < 3; ++elem) {
                        int P;
                        if (disable_srgb_conversion)
                            P = static_cast<int>((*src++) * 255.0f);
                        else
                            P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
                        unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
                        *dst++ = static_cast<unsigned char>(Clamped);
                    }
                }
            }
            break;

        case RT_FORMAT_FLOAT4:
            // This buffer is upside down
            for(int j = height-1; j >= 0; --j) {
                unsigned char *dst = &pix[0] + (3*width*(height-1-j));
                float* src = ((float*)imageData) + (4*width*j);
                for(int i = 0; i < width; i++) {
                    for(int elem = 0; elem < 3; ++elem) {
                        int P;
                        if(disable_srgb_conversion)
                            P = static_cast<int>((*src++) * 255.0f);
                        else
                            P = static_cast<int>(std::pow(*src++, gamma_inv) * 255.0f);
                        unsigned int Clamped = P < 0 ? 0 : P > 0xff ? 0xff : P;
                        *dst++ = static_cast<unsigned char>(Clamped);
                    }

                    // skip alpha
                    src++;
                }
            }
            break;

        default:
            fprintf(stderr, "Unrecognized buffer data type or format.\n");
            exit(2);
            break;
    }

    SavePPM(&pix[0], filename, width, height, 3);

    // Now unmap the buffer
    RT_CHECK_ERROR( rtBufferUnmap(buffer) );
}


void sutil::displayBufferGL( optix::Buffer buffer, bufferPixelFormat format, bool disable_srgb_conversion )
{
    g_image_buffer = buffer->get();
    g_image_buffer_format = format;
    g_disable_srgb_conversion = disable_srgb_conversion;
    displayBuffer();
}


namespace
{

void drawText( const std::string& text, float x, float y, void* font )
{
    // Save state
    glPushAttrib( GL_CURRENT_BIT | GL_ENABLE_BIT );

    glDisable( GL_TEXTURE_2D );
    glDisable( GL_LIGHTING );
    glDisable( GL_DEPTH_TEST);

    static const float3 shadow_color = make_float3( 0.10f );
    glColor3fv( &( shadow_color.x) ); // drop shadow
    // Shift shadow one pixel to the lower right.
    glWindowPos2f(x + 1.0f, y - 1.0f);
    for( std::string::const_iterator it = text.begin(); it != text.end(); ++it )
        glutBitmapCharacter( font, *it );

    static const float3 text_color = make_float3( 0.95f );
    glColor3fv( &( text_color.x) );        // main text
    glWindowPos2f(x, y);
    for( std::string::const_iterator it = text.begin(); it != text.end(); ++it )
        glutBitmapCharacter( font, *it );

    // Restore state
    glPopAttrib();
}

static const float FPS_UPDATE_INTERVAL = 0.5;  //seconds

} // namespace


void sutil::displayFps( unsigned int frame_count )
{
    static double fps = -1.0;
    static unsigned last_frame_count = 0;
    static double last_update_time = sutil::currentTime();
    static double current_time = 0.0;
    current_time = sutil::currentTime();
    if ( current_time - last_update_time > FPS_UPDATE_INTERVAL ) {
        fps = ( frame_count - last_frame_count ) / ( current_time - last_update_time );
        last_frame_count = frame_count;
        last_update_time = current_time;
    }
    if ( frame_count > 0 && fps >= 0.0 ) {
        static char fps_text[32];
        sprintf( fps_text, "fps: %7.2f", fps );
        drawText( fps_text, 10.0f, 10.0f, GLUT_BITMAP_8_BY_13 );
    }
}


void sutil::displayText(const char* text, float x, float y)
{
  drawText(text, x, y, GLUT_BITMAP_8_BY_13);
}


optix::TextureSampler sutil::loadTexture( optix::Context context,
        const std::string& filename, optix::float3 default_color )
{
    bool isHDR = false;
    size_t len = filename.length();
    if(len >= 3) {
      isHDR = (filename[len-3] == 'H' || filename[len-3] == 'h') &&
              (filename[len-2] == 'D' || filename[len-2] == 'd') &&
              (filename[len-1] == 'R' || filename[len-1] == 'r');
    }
    if ( isHDR ) {
        return loadHDRTexture(context, filename, default_color);
    } else {
        return loadPPMTexture(context, filename, default_color);
    }
}


optix::Buffer sutil::loadCubeBuffer( optix::Context context,
        const std::vector<std::string>& filenames )
{
    return loadPPMCubeBuffer( context, filenames );
}


void sutil::calculateCameraVariables( float3 eye, float3 lookat, float3 up,
        float  fov, float  aspect_ratio,
        float3& U, float3& V, float3& W, bool fov_is_vertical )
{
    float ulen, vlen, wlen;
    W = lookat - eye; // Do not normalize W -- it implies focal length

    wlen = length( W );
    U = normalize( cross( W, up ) );
    V = normalize( cross( U, W  ) );

    if ( fov_is_vertical ) {
        vlen = wlen * tanf( 0.5f * fov * M_PIf / 180.0f );
        V *= vlen;
        ulen = vlen * aspect_ratio;
        U *= ulen;
    }
    else {
        ulen = wlen * tanf( 0.5f * fov * M_PIf / 180.0f );
        U *= ulen;
        vlen = ulen / aspect_ratio;
        V *= vlen;
    }
}


void sutil::parseDimensions( const char* arg, int& width, int& height )
{

    // look for an 'x': <width>x<height>
    size_t width_end = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if ( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char *height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        width  = atoi( width_arg );
        height = atoi( height_arg );
        return;
    }

    throw Exception(
            "Failed to parse width, heigh from string '" +
            std::string( arg ) +
            "'" );
}


double sutil::currentTime()
{
#if defined(_WIN32)

    // inv_freq is 1 over the number of ticks per second.
    static double inv_freq;
    static bool freq_initialized = false;
    static BOOL use_high_res_timer = 0;

    if ( !freq_initialized )
    {
        LARGE_INTEGER freq;
        use_high_res_timer = QueryPerformanceFrequency( &freq );
        inv_freq = 1.0/freq.QuadPart;
        freq_initialized = true;
    }

    if ( use_high_res_timer )
    {
        LARGE_INTEGER c_time;
        if( QueryPerformanceCounter( &c_time ) )
            return c_time.QuadPart*inv_freq;
        else
            throw Exception( "sutil::currentTime: QueryPerformanceCounter failed" );
    }

    return static_cast<double>( timeGetTime() ) * 1.0e-3;

#else

    struct timeval tv;
    if( gettimeofday( &tv, 0 ) )
        throw Exception( "sutil::urrentTime(): gettimeofday failed!\n" );

    return  tv.tv_sec+ tv.tv_usec * 1.0e-6;

#endif
}


void sutil::sleep( int seconds )
{
#if defined(_WIN32)
    Sleep( seconds * 1000 );
#else
    ::sleep( seconds );
#endif
}


#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
#define LINE_STR STRINGIFY(__LINE__)

// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR( func )                                  \
  do {                                                             \
    nvrtcResult code = func;                                       \
    if( code != NVRTC_SUCCESS )                                    \
      throw Exception( "ERROR: " __FILE__ "(" LINE_STR "): " +     \
          std::string( nvrtcGetErrorString( code ) ) );            \
  } while( 0 )

static bool readSourceFile( std::string &str, const std::string &filename )
{
    // Try to open file
    std::ifstream file( filename.c_str() );
    if( file.good() )
    {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}

#if CUDA_NVRTC_ENABLED

static void getCuStringFromFile( std::string &cu, std::string& location, const char* sample_name, const char* filename )
{
    std::vector<std::string> source_locations;

    std::string base_dir = std::string( sutil::samplesDir() );

    // Potential source locations (in priority order)
    if( sample_name )
        source_locations.push_back( base_dir + "/" + sample_name + "/" + filename );
    source_locations.push_back( base_dir + "/cuda/" + filename );

    for( std::vector<std::string>::const_iterator it = source_locations.begin(); it != source_locations.end(); ++it ) {
        // Try to get source code from file
        if( readSourceFile( cu, *it ) )
        {
            location = *it;
            return;
        }

    }

    // Wasn't able to find or open the requested file
    throw Exception( "Couldn't open source file " + std::string( filename ) );
}

static std::string g_nvrtcLog;

static void getPtxFromCuString( std::string &ptx, const char* sample_name, const char* cu_source, const char* name, const char** log_string )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char *> options;

    std::string base_dir = std::string( sutil::samplesDir() );

    // Set sample dir as the primary include path
    std::string sample_dir;
    if( sample_name )
    {
        sample_dir = std::string( "-I" ) + base_dir + "/" + sample_name;
        options.push_back( sample_dir.c_str() );
    }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char *abs_dirs[] = { SAMPLES_ABSOLUTE_INCLUDE_DIRS };
    const char *rel_dirs[] = { SAMPLES_RELATIVE_INCLUDE_DIRS };

    const size_t n_abs_dirs = sizeof( abs_dirs ) / sizeof( abs_dirs[0] );
    for( size_t i = 0; i < n_abs_dirs; i++ )
        include_dirs.push_back(std::string( "-I" ) + abs_dirs[i]);
    const size_t n_rel_dirs = sizeof( rel_dirs ) / sizeof( rel_dirs[0] );
    for( size_t i = 0; i < n_rel_dirs; i++ )
        include_dirs.push_back(std::string( "-I" ) + base_dir + rel_dirs[i]);
    for( std::vector<std::string>::const_iterator it = include_dirs.begin(); it != include_dirs.end(); ++it )
        options.push_back( it->c_str() );

    // Collect NVRTC options
    const char *compiler_options[] = { CUDA_NVRTC_OPTIONS };
    const size_t n_compiler_options = sizeof( compiler_options ) / sizeof( compiler_options[0] );
    for( size_t i = 0; i < n_compiler_options - 1; i++ )
        options.push_back( compiler_options[i] );

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int) options.size(), options.data() );

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw Exception( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &ptx_size ) );
    ptx.resize( ptx_size );
    NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &ptx[0] ) );

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}

#else // CUDA_NVRTC_ENABLED

static void getPtxStringFromFile( std::string &ptx, const char* sample_name, const char* filename )
{
    std::string source_filename;
    if (sample_name)
        source_filename = std::string( sutil::samplesPTXDir() ) + "/" + std::string( sample_name ) + "_generated_" + std::string( filename ) + ".ptx";
    else
        source_filename = std::string( sutil::samplesPTXDir() ) + "/cuda_compile_ptx_generated_" + std::string( filename ) + ".ptx";

    // Try to open source PTX file
    if (!readSourceFile( ptx, source_filename ))
        throw Exception( "Couldn't open source file " + source_filename );
}

#endif // CUDA_NVRTC_ENABLED

struct PtxSourceCache
{
    std::map<std::string, std::string *> map;
    ~PtxSourceCache()
    {
        for( std::map<std::string, std::string *>::const_iterator it = map.begin(); it != map.end(); ++it )
            delete it->second;
    }
	void Clear() { 
		for (std::map<std::string, std::string *>::const_iterator it = map.begin(); it != map.end(); ++it)
			delete it->second;
		map.clear();
	};
};
static PtxSourceCache g_ptxSourceCache;

const char* sutil::getPtxString(
    const char* sample,
    const char* filename,
    const char** log )
{
    if (log)
        *log = NULL;

    std::string *ptx, cu;
    std::string key = std::string( filename ) + ";" + ( sample ? sample : "" );
    std::map<std::string, std::string *>::iterator elem = g_ptxSourceCache.map.find( key );

    if( elem == g_ptxSourceCache.map.end() )
    {
        ptx = new std::string();
#if CUDA_NVRTC_ENABLED
        std::string location;
        getCuStringFromFile( cu, location, sample, filename );
        getPtxFromCuString( *ptx, sample, cu.c_str(), location.c_str(), log );
#else
        getPtxStringFromFile( *ptx, sample, filename );
#endif
        g_ptxSourceCache.map[key] = ptx;
    }
    else
    {
        ptx = elem->second;
    }

    return ptx->c_str();
}

void sutil::ClearPtxCache() {
	g_ptxSourceCache.Clear();
}

void sutil::ensureMinimumSize(int& w, int& h)
{
    if (w <= 0) w = 1;
    if (h <= 0) h = 1;
}

void sutil::ensureMinimumSize(unsigned& w, unsigned& h)
{
    if (w == 0) w = 1;
    if (h == 0) h = 1;
}
