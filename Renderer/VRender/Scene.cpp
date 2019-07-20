#include "Scene.hpp"

namespace VRender {
	string VScene::name;
	vector<string> VScene::split(const string& str, const string& delim) {
		vector<string> res;
		if ("" == str) return res;
		//先将要切割的字符串从string类型转换为char*类型  
		char* strs = new char[str.length() + 1]; //不要忘了  
		strcpy(strs, str.c_str());

		char* d = new char[delim.length() + 1];
		strcpy(d, delim.c_str());

		char* p = strtok(strs, d);
		while (p) {
			string s = p; //分割得到的字符串转换为string类型  
			res.push_back(s); //存入结果数组  
			p = strtok(NULL, d);
		}

		return res;
	}
	void VScene::ProcessObject(const string& data) {
		auto parts = split(data, "|");

		VObject object = VObjectManager::CreateNewObject();
		object->SetComponent(VComponents::Create<VMeshRenderer>());
		object->SetComponent(VComponents::Create<VMeshFilter>());

		for (int i = 0; i < 4; i++)
		{
			string str;
			if (i >= parts.size()) {
				if (i == 2) str = "(1,1,1)(0,0,0)(1,1,1)";
				else str = "default";
			}
			else {
				str = parts[i];
			}
			switch (i)
			{
			case 0:
				object->name = str;
				break;
			case 1:
				object->MeshFilter()->SetMesh(VResources::Find<VMesh>(str));
				break;
			case 2:
			{
				float px, py, pz, rx, ry, rz, sx, sy, sz;
				int err = sscanf(str.c_str(), "(%f,%f,%f)(%f,%f,%f)(%f,%f,%f)", &px, &py, &pz, &rx, &ry, &rz, &sx, &sy, &sz);
				if (err != 9) throw Exception("Transform foramt error");
				*object->Transform()->Position<float3>() = make_float3(px, py, pz);
				*object->Transform()->Rotation<float3>() = make_float3(rx, ry, rz);
				*object->Transform()->Scale<float3>() = make_float3(sx, sy, sz);
				break;
			}
			case 3:
				object->MeshRenderer()->SetMaterial(VResources::Find<VMaterial>(str));
				break;
			default:
				break;
			}
		}
	}
	void VScene::ProcessLight(const string& data) {
		auto parts = split(data, "|");

		int type = prime::VLightObj::Type(parts[1][0] - '0');
		string mat = parts[3];

		int id = VLightManager::CreateLight(type, mat);
		VLight light = VLightManager::GetLight(id);

		light->name = parts[0];

		float px, py, pz, rx, ry, rz, sx, sy, sz, cx, cy, cz;
		int err = sscanf(parts[2].c_str(), "(%f,%f,%f)(%f,%f,%f)(%f,%f,%f)(%f,%f,%f)", &px, &py, &pz, &rx, &ry, &rz, &sx, &sy, &sz, &cx, &cy, &cz);
		if (err != 12) throw Exception("Transform foramt error");

		light->position = make_float3(px, py, pz);
		light->rotation = make_float3(rx, ry, rz);
		light->scale = make_float3(sx, sy, sz);
		light->emission = make_float3(cx, cy, cz);
	}
	void VScene::ProcessCamera(const string& data) {

		VCamera cam = VRenderer::Instance().Camera();

		float px, py, pz, rx, ry, rz, sx, sy, sz, fx, fy;
		int err = sscanf(data.c_str(), "(%f,%f,%f)(%f,%f,%f)(%f,%f,%f)(%f,%f)", &px, &py, &pz, &rx, &ry, &rz, &sx, &sy, &sz, &fx, &fy);
		if (err != 11) throw Exception("Camera foramt error");
		cam->position = make_float3(px, py, pz);
		cam->forward = normalize(make_float3(rx, ry, rz));
		cam->up = normalize(make_float3(sx, sy, sz));
		cam->right = cross(cam->forward, cam->up);
		cam->fov = make_float2(fx, fy);
		cam->dirty = true;
	}
}