#include <GlutEvent.h>

//#define QT

#ifdef QT

#include <QApplication>
#include <QWindow>

#include <ui_renderer_window.h>

class UserInterface {
private:
	UserInterface() {};
	UserInterface(const UserInterface&) abandon;
	UserInterface& operator=(const UserInterface&) abandon;
public:
	static UserInterface& Instance() {
		static UserInterface& instance = UserInterface();
		return instance;
	}
	Ui::RendererWindow userInterface;

	inline static Ui::RendererWindow& RendererWindow() {
		return Instance().userInterface;
	}

};

int main(int argc, char *argv[])
{
	int a;
	char** b = NULL;
	QApplication c(a, b);
	auto window = new QMainWindow;
	UserInterface::RendererWindow().setupUi(window);
	window->show();
	window->setWindowTitle("VRenderer");
	UserInterface::RendererWindow().p0->setText("aaa");



	return c.exec();
}

#else


#include <GUI.hpp>
#include <thread>
#include <shellapi.h>
using namespace std;

int main(int argc, char** argv) {
	//HWND hwnd;
	//hwnd = FindWindow("ConsoleWindowClass", NULL);
	//if (hwnd)
	//{
	//	ShowOwnedPopups(hwnd, SW_HIDE);
	//	ShowWindow(hwnd, SW_HIDE);
	//}
	thread([&]() {
		try
		{
			glutInitialize(&argc, argv);

			glewInit();

			OptiXLayer::LazyInit();
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

#endif // 1



