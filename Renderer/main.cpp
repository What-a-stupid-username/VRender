#define TINYOBJLOADER_IMPLEMENTATION

#include <VRender.hpp>
#include "default_pipeline.hpp"


#include "GUI/renderView.hpp"
#include "GUI/DXWrapper.hpp"


using namespace std;
using namespace VRender;



int main(int argc, char** argv) {

	#pragma region Hide console
	#ifndef _DEBUG
	HWND hwnd;
	hwnd = FindWindow("ConsoleWindowClass", NULL);
	if (hwnd)
	{
		ShowOwnedPopups(hwnd, SW_HIDE);
		ShowWindow(hwnd, SW_HIDE);
	}
	#endif // !_DEBUG
	#pragma endregion



	struct Helper
	{
		static void CreateCube(float3 position, float3 rotation, float3 scale, string material = "default") {
			VMesh mesh = VResources::Find<VMesh>("Cube.obj");

			VMeshFilter filter = VComponents::Create<VMeshFilter>();

			filter->SetMesh(mesh);

			VMaterial mat = VResources::Find<VMaterial>(material);

			VMeshRenderer render = VComponents::Create<VMeshRenderer>();

			render->SetMaterial(mat);

			VObject obj = VObjectManager::CreateNewObject();

			obj->SetComponent(filter);
			obj->SetComponent(render);

			*obj->Transform()->Position<float3>() = position;
			*obj->Transform()->Rotation<float3>() = rotation;
			*obj->Transform()->Scale<float3>() = scale;
		}
	};


	Helper::CreateCube(make_float3(0, -1, 0), make_float3(0, 0, 0), make_float3(1, 1, 1), "default");

	VLightManager::CreateLight();

	VRenderer& renderer = VRenderer::Instance();
	renderer.SetupPipeline<DefaultPipeline>();



	#pragma region Show up GUI views
	thread([&]() {
		try
		{
			glutInitialize(&argc, argv);

			glewInit();
			glutRun();

			return 0;
		}
		catch (Exception & e) {
			cout << e.getErrorString() << endl;
			system("PAUSE");
		}
	}).detach();

	int stat = Main_Loop(argc, argv);
	VRenderer::Instance().Join();
	return stat;
	#pragma endregion
}