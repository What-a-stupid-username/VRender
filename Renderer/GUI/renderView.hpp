#pragma once

#include "VRender/VRender.hpp"
#include "resource.h"

#include <math.h>


class GetDistanceOf2linesIn3D
{
public:
	void SetLineA(double A1x, double A1y, double A1z, double A2x, double A2y, double A2z)
	{
		a1_x = A1x;
		a1_y = A1y;
		a1_z = A1z;

		a2_x = A2x;
		a2_y = A2y;
		a2_z = A2z;
	}

	void SetLineB(double B1x, double B1y, double B1z, double B2x, double B2y, double B2z)
	{
		b1_x = B1x;
		b1_y = B1y;
		b1_z = B1z;

		b2_x = B2x;
		b2_y = B2y;
		b2_z = B2z;
	}

	void GetDistance()
	{
		double d1_x = a2_x - a1_x;
		double d1_y = a2_y - a1_y;
		double d1_z = a2_z - a1_z;

		double d2_x = b2_x - b1_x;
		double d2_y = b2_y - b1_y;
		double d2_z = b2_z - b1_z;

		double e_x = b1_x - a1_x;
		double e_y = b1_y - a1_y;
		double e_z = b1_z - a1_z;


		double cross_e_d2_x, cross_e_d2_y, cross_e_d2_z;
		cross(e_x, e_y, e_z, d2_x, d2_y, d2_z, cross_e_d2_x, cross_e_d2_y, cross_e_d2_z);
		double cross_e_d1_x, cross_e_d1_y, cross_e_d1_z;
		cross(e_x, e_y, e_z, d1_x, d1_y, d1_z, cross_e_d1_x, cross_e_d1_y, cross_e_d1_z);
		double cross_d1_d2_x, cross_d1_d2_y, cross_d1_d2_z;
		cross(d1_x, d1_y, d1_z, d2_x, d2_y, d2_z, cross_d1_d2_x, cross_d1_d2_y, cross_d1_d2_z);

		double t1, t2;
		t1 = dot(cross_e_d2_x, cross_e_d2_y, cross_e_d2_z, cross_d1_d2_x, cross_d1_d2_y, cross_d1_d2_z);
		t2 = dot(cross_e_d1_x, cross_e_d1_y, cross_e_d1_z, cross_d1_d2_x, cross_d1_d2_y, cross_d1_d2_z);
		double dd = norm(cross_d1_d2_x, cross_d1_d2_y, cross_d1_d2_z);
		t1 /= dd * dd;
		t2 /= dd * dd;

		//得到最近的位置
		PonA_x = (a1_x + (a2_x - a1_x) * t1);
		PonA_y = (a1_y + (a2_y - a1_y) * t1);
		PonA_z = (a1_z + (a2_z - a1_z) * t1);

		PonB_x = (b1_x + (b2_x - b1_x) * t2);
		PonB_y = (b1_y + (b2_y - b1_y) * t2);
		PonB_z = (b1_z + (b2_z - b1_z) * t2);

		distance = norm(PonB_x - PonA_x, PonB_y - PonA_y, PonB_z - PonA_z);
	}



	double PonA_x;//两直线最近点之A线上的点的x坐标
	double PonA_y;//两直线最近点之A线上的点的y坐标
	double PonA_z;//两直线最近点之A线上的点的z坐标
	double PonB_x;//两直线最近点之B线上的点的x坐标
	double PonB_y;//两直线最近点之B线上的点的y坐标
	double PonB_z;//两直线最近点之B线上的点的z坐标
	double distance;//两直线距离
private:
	//直线A的第一个点
	double a1_x;
	double a1_y;
	double a1_z;
	//直线A的第二个点
	double a2_x;
	double a2_y;
	double a2_z;

	//直线B的第一个点
	double b1_x;
	double b1_y;
	double b1_z;

	//直线B的第二个点
	double b2_x;
	double b2_y;
	double b2_z;


	//点乘
	double dot(double ax, double ay, double az, double bx, double by, double bz) { return ax * bx + ay * by + az * bz; }
	//向量叉乘得到法向量，最后三个参数为输出参数
	void cross(double ax, double ay, double az, double bx, double by, double bz, double& x, double& y, double& z)
	{
		x = ay * bz - az * by;
		y = az * bx - ax * bz;
		z = ax * by - ay * bx;
	}
	//向量取模
	double norm(double ax, double ay, double az) { return sqrt(dot(ax, ay, az, ax, ay, az)); }
};

GetDistanceOf2linesIn3D getDistanceOf2linesIn3D;

float3 GetNearestPointOfTwoRay(float3 pos0, float3 dir0, float3 pos1, float3 dir1) {
	dir0 += pos0; dir1 += pos1;
	getDistanceOf2linesIn3D.SetLineA(pos0.x, pos0.y, pos0.z, dir0.x, dir0.y, dir0.z);
	getDistanceOf2linesIn3D.SetLineB(pos1.x, pos1.y, pos1.z, dir1.x, dir1.y, dir1.z);
	getDistanceOf2linesIn3D.GetDistance();
	return make_float3(getDistanceOf2linesIn3D.PonA_x, getDistanceOf2linesIn3D.PonA_y, getDistanceOf2linesIn3D.PonA_z);
}
float3 GetRayDirFromScreenPoint(float2 pos) {
	auto& renderer = VRender::VRenderer::Instance();
	VRender::VCamera cam = renderer.Camera();
	float2 fov = make_float2(tan(cam->fov.x / 180 * 3.14159265389), tan(cam->fov.y / 180 * 3.14159265389));
	float2 d = pos * 2.f - 1.f;
	float3 ray_origin = cam->position;
	return normalize(d.x * cam->right * fov.y + d.y * cam->up * fov.x + cam->forward);
}


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
				
				if (handle_type == 1) {
					float3 axle = make_float3(0, 0, 0);
					((float*)& axle)[handle_axle - 1] = 1;
					float3 point0 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_ori);
					float3 point1 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_now);
					float3 bias = point1 - point0;
					bias *= axle;
					renderer.Lock();
					*trans->Position<float3>() += bias;
					trans->MarkDirty();
					renderer.Unlock();
				}
				else if (handle_type == 2) {

				}
				else if (handle_type == 3) {
					float3 axle = make_float3(0, 0, 0);
					((float*)& axle)[handle_axle - 1] = 1;
					float3 point0 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_ori);
					float3 point1 = GetNearestPointOfTwoRay(obj_pos, axle, cam_pos, ray_dir_now);
					float3 bias = point1 - point0;
					bias *= axle;
					renderer.Lock();
					*trans->Scale<float3>() += bias;
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