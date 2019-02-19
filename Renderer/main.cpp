#include <QApplication>
#include <QWindow>

#define QT

#ifdef QT

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


#include <GlutEvent.h>

int main(int argc, char** argv) {
	int SIZE = 32;
	try {
		Context context = Context::create();

		context->setRayTypeCount(1);
		context->setEntryPointCount(1);
		context->setStackSize(1800);
		context["length"]->setUint(0);

		// Setup programs
		const char *ptx = sutil::getPtxString("Renderer", "bitonic_sort.cu");
		context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "sort"));
		context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
		context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));

		Buffer length_buffer;
		length_buffer = context->createBuffer(RTbuffertype::RT_BUFFER_INPUT, RTformat::RT_FORMAT_INT, 3);
		context["length_buffer"]->set(length_buffer);

		Buffer input_buffer;
		input_buffer = context->createBuffer(RTbuffertype::RT_BUFFER_INPUT, RTformat::RT_FORMAT_INT, SIZE);
		context["intput_buffer"]->set(input_buffer);
		Buffer output_buffer;
		output_buffer = context->createBuffer(RTbuffertype::RT_BUFFER_OUTPUT, RTformat::RT_FORMAT_USER, SIZE);
		output_buffer->setElementSize(sizeof(float) * 4);
		context["output_buffer"]->set(output_buffer);

		auto length = (int*)length_buffer->map();
		length[0] = 0; length[1] = 0; length[2] = 0;
		length_buffer->unmap();

		auto data = (int*)input_buffer->map();
		for (int i = 0; i < SIZE; i++)
		{
			data[i] = rand()%10000 + 1;
		}
		input_buffer->unmap();
		CommandList cb = context->createCommandList();
		for (int i = 0; i < 1; i++)
		{
			cb->appendLaunch(0, SIZE, 1);
		}
		cb->finalize();
		float t = sutil::currentTime();
		cb->execute();
		t = sutil::currentTime() - t;

		auto res = (int*)output_buffer->map();
		map<int,bool> vali;
		for (int i = 0; i < SIZE; i++)
		{
			cout << res[i] << endl;
		}
		output_buffer->unmap();
		cout << t / 10;
	}
	catch (Exception& e) {
		cout << e.getErrorString() << endl;
	}
	
	system("PAUSE");

//	try
//	{
//		glutInitialize(&argc, argv);
//
//#ifndef __APPLE__
//		glewInit();
//#endif
//		OptiXLayer::LazyInit();
//		OptiXLayer::LoadScene();
//
//		glutRun();
//
//		return 0;
//	}
//	catch (Exception& e) {
//		cout << e.getErrorString() << endl;
//		system("PAUSE");
//	}
}

#endif // 1



