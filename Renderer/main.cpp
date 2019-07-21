#define TINYOBJLOADER_IMPLEMENTATION

#include <VRender/VRender.hpp>
#include "default_pipeline.hpp"
#include "VRender/Scene.hpp"

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

	
	VScene::LoadScene(string(sutil::samplesDir()) + "/Cornell");

	VRenderer& renderer = VRenderer::Instance();
	renderer.SetupPipeline<DefaultPipeline>();
	//renderer.EnableRenderer(false);


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