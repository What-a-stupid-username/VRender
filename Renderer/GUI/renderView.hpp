#pragma once

#include "VRender/VRender.hpp"
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
	else if (state == GLUT_UP)
	{
		mouse_button = -1;
	}
}


float w_width, w_height;

void glutMouseMotion(int x, int y)
{
	int mod = glutGetModifiers();

	const float2 from = { static_cast<float>(mouse_prev_pos.x),
	static_cast<float>(mouse_prev_pos.y) };
	const float2 to = { static_cast<float>(x),
		static_cast<float>(y) };

	const float2 a = { from.x / w_width, from.y / w_height };
	const float2 b = { to.x / w_width, to.y / w_height };
	float2 del = b - a;
	auto cam = VRender::VRenderer::Instance().Camera();

	if (mod == GLUT_ACTIVE_ALT) {
		if (mouse_button == GLUT_LEFT_BUTTON) //绕视点中心旋转
		{
			float3 pos = cam->position;
			float3 forward = cam->forward;
			float3 up = cam->up;
			float3 ori = pos + forward * 5;
			float3 ri = cam->right;

			pos += (del.y * up - del.x * ri) * 10;
			forward = normalize(ori - pos);
			pos = forward * -5 + ori;
			ri = cross(forward, float3{ 0,1,0 });
			up = cross(ri, forward);
			VRender::VRenderer::Instance().Lock();
			cam->position = pos;
			cam->forward = forward;
			cam->up = up;
			cam->right = ri;
			cam->dirty = true;
			VRender::VRenderer::Instance().Unlock();
		}
		else if (mouse_button == GLUT_RIGHT_BUTTON) //缩放
		{
			VRender::VRenderer::Instance().Lock();
			cam->position += cam->forward * del.y * 10;
			cam->dirty = true;
			VRender::VRenderer::Instance().Unlock();

		}
		else //平移
		{
			VRender::VRenderer::Instance().Lock();
			cam->position += (cam->right * -del.x + cam->up * del.y) * 10;
			cam->dirty = true;
			VRender::VRenderer::Instance().Unlock();
		}
	}
	else 
	{
		if (mouse_button == GLUT_RIGHT_BUTTON) //绕视点旋转
		{
			float3 forward = cam->forward;
			float3 up = cam->up;
			float3 ri = cam->right;

			forward = normalize(forward + (cam->right * -del.x + cam->up * del.y) * 2);
			ri = cross(forward, float3{ 0,1,0 });
			up = cross(ri, forward);
			VRender::VRenderer::Instance().Lock();
			cam->forward = forward;
			cam->up = up;
			cam->right = ri;
			cam->dirty = true;
			VRender::VRenderer::Instance().Unlock();
		}
		else if (mouse_button == GLUT_LEFT_BUTTON) //选取
		{

		}
		else//平移
		{		
			VRender::VRenderer::Instance().Lock();
			cam->position += (cam->right * -del.x + cam->up * del.y) * 10;
			cam->dirty = true;
			VRender::VRenderer::Instance().Unlock();
		}
	}

	mouse_prev_pos = make_int2(x, y);
}

int buffer = 0;

void glutKeyboardPress(unsigned char k, int x, int y)
{

}

void glutResize(int w, int h)
{
		glViewport(0, 0, w, h);
		w_width = w, w_height = h;
		glutPostRedisplay();
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
	
	sutil::displayBufferGL(VRender::VRenderer::Instance().GetRenderResult());

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
	Style = Style & ~WS_MAXIMIZEBOX;
	::SetWindowLongPtr(GL_Window, GWL_STYLE, Style);
	HINSTANCE hInstance = ::GetModuleHandle(NULL);
	if (NULL != hInstance) {
		HICON hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
		::SendMessage(GL_Window, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
		::SendMessage(GL_Window, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
	}
	glutHideWindow();
}

void glutExitProgram() {

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