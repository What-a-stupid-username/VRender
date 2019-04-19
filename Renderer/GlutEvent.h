#pragma once

#include "CommonInclude.h"
#include "OptiXLayer.h"
#include "resource.h"

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
	
}

void glutResize(int w, int h)
{
	if (OptiXLayer::ResizeBuffer(w, h)) {
		glViewport(0, 0, w, h);
		glutPostRedisplay();
	}
}


HWND GL_Window;
HWND DX_Window;
void glutDisplay()
{
	static bool first_run = true;
	if (first_run) {
		SetForegroundWindow(GL_Window);
		SetFocus(GL_Window);
		first_run = false;
	}
	OptiXLayer::RenderResult();

	OptiXLayer::Lock();
	sutil::displayBufferGL(OptiXLayer::GetResult());
	OptiXLayer::Unlock();

	glutSwapBuffers();

	RECT rect, rect2;
	GetWindowRect(GL_Window, &rect);
	GetWindowRect(DX_Window, &rect2);
	MoveWindow(GL_Window, rect2.left - (rect.right - rect.left), rect2.top, rect.right - rect.left, rect.bottom - rect.top, FALSE);
}

void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(512, 512);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("VRenderer");
	GL_Window = GetActiveWindow();
	LONG_PTR Style = ::GetWindowLongPtr(GL_Window, GWL_STYLE);
	Style = Style &~WS_MAXIMIZEBOX;
	::SetWindowLongPtr(GL_Window, GWL_STYLE, Style);
	HINSTANCE hInstance = ::GetModuleHandle(NULL);
	if (NULL != hInstance) 	{
		HICON hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
		::SendMessage(GL_Window, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
		::SendMessage(GL_Window, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
	}
	glutHideWindow();
}

void glutExitProgram() {
	OptiXLayer::Release();
	exit(0);
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
	glutCloseFunc(glutExitProgram);
#else
	atexit(OptiXLayer::Release);
#endif

	glutMainLoop();
}