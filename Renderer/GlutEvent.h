#pragma once

#include "CommonInclude.h"
#include "OptiXLayer.h"

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

//helper
sutil::Arcball arcball;

void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
}


void glutMouseMotion(int x, int y)
{
	float width, height;
	OptiXLayer::GetBufferSize(width, height);
	Camera& camera = OptiXLayer::Camera();
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		camera.Scale(scale);
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
			static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
			static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera.Rotate(arcball.rotate(b, a));
	}
	mouse_prev_pos = make_int2(x, y);
}

int buffer = 0;

void glutKeyboardPress(unsigned char k, int x, int y)
{
	switch (k)
	{
	case('s'):
	{
		const std::string outputImage = "screenshot.ppm";
		std::cerr << "Saving current frame to '" << outputImage << "'\n";
		sutil::displayBufferPPM(outputImage.c_str(), OptiXLayer::OriginBuffer(), false);
		break;
	}
	case('w'):
	{
		OptiXLayer::ClearScene();
		OptiXLayer::LoadScene();
		break;
	}
	case('b'):
	{
		buffer++;
		buffer %= 3;
		break;
	}
	case('o'):
	{
		OptiXLayer::ChangeExposure(-0.02);
		break;
	}
	case('p'):
	{
		OptiXLayer::ChangeExposure(0.02);
		break;
	}
	}
}

void glutResize(int w, int h)
{
	if (OptiXLayer::ResizeBuffer(w, h)) {
		glViewport(0, 0, w, h);
		glutPostRedisplay();
	}
}


void glutDisplay()
{
	OptiXLayer::RenderResult();

	switch (buffer)
	{
	case 0:
		sutil::displayBufferGL(OptiXLayer::OriginBuffer());
		break;
	case 1:
		sutil::displayBufferGL(OptiXLayer::ToneMappedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
		break;
	case 2:
		sutil::displayBufferGL(OptiXLayer::DenoisedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
		break;
	}

	{
		int k = OptiXLayer::Camera().StaticFrameNum();
		sutil::displayFps(k);
		static char text_[32];
		sprintf(text_, "frame: %d", k);
		string text = string(text_);
		sutil::displayText(text.c_str(), 200, 10);
	}

	glutSwapBuffers();
}


void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(512, 512);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("VRenderer");
	glutHideWindow();
}


void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, 512, 512);

	glutShowWindow();
	glutReshapeWindow(512, 512);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(OptiXLayer::Release);
#else
	atexit(OptiXLayer::Release);
#endif

	glutMainLoop();
}