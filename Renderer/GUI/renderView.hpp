#pragma once

#include "VRender/VRender.hpp"
#include "resource.h"
#include "HandleMathHelper.hpp"


// Mouse state
int2	mouse_prev_pos;
int2	mouse_press_pos;
bool	moved_since_pressed;
int		mouse_button;

float w_width, w_height;

int handle_axle, handle_type;

void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
		mouse_press_pos = make_int2(x, y);
		moved_since_pressed = false;
		if (button == GLUT_LEFT_BUTTON) {
			VRender::VRenderer& renderer = VRender::VRenderer::Instance();
			int selected_id = renderer.GetSelectedObject();
			if (selected_id != -1) {
				optix::Buffer id_mask = renderer.GetIDMask();
				renderer.Lock();
				RTsize w, h; id_mask->getSize(w, h);
				int click_id = ((int*)id_mask->map())[(h - y) * w + x];
				id_mask->unmap();
				renderer.Unlock();
				if (click_id < -1) {
					click_id = -click_id;
					handle_axle = click_id / 10;
					handle_type = click_id % 10;
					renderer.EnableHandle(false);
				}
			}
		}
	}
	else if (state == GLUT_UP)
	{
		handle_axle = 0, handle_type = 0;
		mouse_button = -1;
		VRender::VRenderer& renderer = VRender::VRenderer::Instance();
		renderer.EnableHandle(true);
		if (!moved_since_pressed) {
			if (button == GLUT_LEFT_BUTTON) {
				optix::Buffer id_mask = renderer.GetIDMask();
				renderer.Lock();
				RTsize w, h; id_mask->getSize(w, h);
				int click_id = ((int*)id_mask->map())[(h - y) * w + x];
				id_mask->unmap();
				renderer.Unlock();
				if (click_id >= -1) {
					renderer.SetSelectedObject(click_id);
				}
			}
		}
	}
}



void glutMouseMotion(int x, int y)
{
	if (abs(x - mouse_press_pos.x) + abs(y - mouse_press_pos.y) > 10) moved_since_pressed = true;
	int mod = glutGetModifiers();

	const float2 from = { static_cast<float>(mouse_prev_pos.x),
	static_cast<float>(mouse_prev_pos.y) };
	const float2 to = { static_cast<float>(x),
		static_cast<float>(y) };

	const float2 a = { from.x / w_width, (w_height - from.y) / w_height };
	const float2 b = { to.x / w_width, (w_height - to.y) / w_height };
	const float2 o = { (float)mouse_press_pos.x / w_width, float(w_height - mouse_press_pos.y) / w_height };
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

			pos -= (del.y * up + del.x * ri) * 10;
			forward = normalize(ori - pos);
			pos = forward * -5 + ori;
			ri = cross(forward, float3{ 0,1,0 });
			up = cross(ri, forward);
			cam->position = pos;
			cam->forward = forward;
			cam->up = up;
			cam->right = ri;
			cam->dirty = true;
		}
		else if (mouse_button == GLUT_RIGHT_BUTTON) //缩放
		{
			cam->position -= cam->forward * del.y * 10;
			cam->dirty = true;

		}
		else //平移
		{
			cam->position -= (cam->right * del.x + cam->up * del.y) * 10;
			cam->dirty = true;
		}
	}
	else 
	{
		if (mouse_button == GLUT_RIGHT_BUTTON) //绕视点旋转
		{
			float3 forward = cam->forward;
			float3 up = cam->up;
			float3 ri = cam->right;

			forward = normalize(forward - (cam->right * del.x + cam->up * del.y) * 2);
			ri = cross(forward, float3{ 0,1,0 });
			up = cross(ri, forward);
			cam->forward = forward;
			cam->up = up;
			cam->right = ri;
			cam->dirty = true;
		}
		else if (mouse_button == GLUT_LEFT_BUTTON) //选取
		{
			if (handle_axle != 0) {
				auto& renderer = VRender::VRenderer::Instance();
				auto& selected_obect = VRender::VObjectManager::GetObjectByID(renderer.GetSelectedObject());
				VRender::VTransform trans = selected_obect->Transform();
				float3 obj_pos = *trans->Position<float3>();
				float3 cam_pos = renderer.Camera()->position;
				float3 ray_dir_ori = GetRayDirFromScreenPoint(a);
				float3 ray_dir_now = GetRayDirFromScreenPoint(b);
				float3 axle = make_float3(0, 0, 0);
				((float*)& axle)[handle_axle - 1] = 1;

				if (handle_type == 1) {
					float3 point0 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_ori);
					float3 point1 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_now);
					float3 bias = point1 - point0;
					bias *= axle;
					renderer.Lock();
					if (selected_obect->light_id != -1) {
						VRender::VLightManager::GetLight(selected_obect->light_id)->position += bias;
						VRender::VLightManager::MarkDirty(selected_obect->light_id);
					}
					else {
						*trans->Position<float3>() += bias;
						trans->MarkDirty();
					}
					renderer.Unlock();
				}
				else if (handle_type == 2) {
					float3 ray_dir = GetRayDirFromScreenPoint(o);
					float3 point0 = GetNearestPointOfSpehereByRay(obj_pos, cam_pos, ray_dir) - obj_pos;
					float3 tangent =  cross(normalize(point0), axle);
					float2 tanget_screen = normalize(make_float2(dot(tangent, cam->right), dot(tangent, cam->up)));
					float3 bias = dot(del, tanget_screen) * axle * -100;
					renderer.Lock();
					if (selected_obect->light_id != -1) {
						VRender::VLightManager::GetLight(selected_obect->light_id)->rotation += bias;
						VRender::VLightManager::MarkDirty(selected_obect->light_id);
					}
					else {
						*trans->Rotation<float3>() += bias;
						trans->MarkDirty();
					}
					renderer.Unlock();
				}
				else if (handle_type == 3) {
					float3 point0 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_ori);
					float3 point1 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_now);
					float3 bias = point1 - point0;
					bias *= axle;
					renderer.Lock();
					if (selected_obect->light_id != -1) {
						VRender::VLightManager::GetLight(selected_obect->light_id)->scale += bias;
						VRender::VLightManager::MarkDirty(selected_obect->light_id);
					}
					else {
						*trans->Scale<float3>() += bias;
						trans->MarkDirty();
					}
					trans->MarkDirty();
					renderer.Unlock();
				}
			}
		}
		else//平移
		{		
			cam->position -= (cam->right * del.x + cam->up * del.y) * 10;
			cam->dirty = true;
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
	
	sutil::displayBufferGL(VRender::VRenderer::Instance().GetRenderTarget());

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