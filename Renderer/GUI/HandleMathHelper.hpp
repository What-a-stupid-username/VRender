#pragma once
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

float3 GetNearestPointOfSpehereByRay(float3 pos0, float3 pos1, float3 dir) {
	float3 del_pos = pos1 - pos0;
	float m = -(del_pos.x * dir.x + del_pos.y * dir.y + del_pos.z * dir.z) / (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
	float3 pos = pos1 + dir * m;
	return pos;
}