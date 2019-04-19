#include <GlutEvent.h>

#include <GUI.hpp>
#include <thread>
#include <shellapi.h>

#include <Support/Components.h>

using namespace std;

int main(int argc, char** argv) {
	//HWND hwnd;
	//hwnd = FindWindow("ConsoleWindowClass", NULL);
	//if (hwnd)
	//{
	//	ShowOwnedPopups(hwnd, SW_HIDE);
	//	ShowWindow(hwnd, SW_HIDE);
	//}

	OptiXLayer::LazyInit();

	thread([&]() {
		try
		{
			glutInitialize(&argc, argv);

			glewInit();

			OptiXLayer::LoadScene();

			glutRun();

			return 0;
		}
		catch (Exception& e) {
			cout << e.getErrorString() << endl;
			system("PAUSE");
		}
	}).detach();

	return Main_Loop(argc, argv);
}