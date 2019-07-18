#include "glviewer.h"
#include <QFileDialog>
#include <QColorDialog>
#include <thread>



GLViewer::GLViewer(QWidget *parent)
	:QGLWidget(parent)
{
	
}

GLViewer::~GLViewer()
{
	
}

void GLViewer::mousePressEvent(QMouseEvent *event_)
{
	mouse_button = event_->button();
	mouse_prev_pos = make_int2(event_->x(), event_->y());
	
}

void GLViewer::mouseReleaseEvent(QMouseEvent *event)
{
	
}

void GLViewer::mouseMoveEvent(QMouseEvent *event_)
{
	float x = event_->x();
	float y = event_->y();
	float width, height;
	OptiXLayer::GetBufferSize(width, height);
	Camera& camera = OptiXLayer::Camera();
	if (mouse_button == 2)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = std::min<float>(dmax, 0.9f);
		camera.Scale(scale);
	}
	else if (mouse_button == 1)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
			static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
			static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera.Rotate(arcball.rotate(b, a));
	}
	mouse_prev_pos = make_int2(x, y);
}

void GLViewer::wheelEvent(QWheelEvent *event)
{
	
}

void GLViewer::paintGL()
{
	OptiXLayer::RenderResult();
	switch (buffer)
	{
	case 0:
		sutil::displayBufferGL(OptiXLayer::OriginBuffer());
		break;
	case 1:
		sutil::displayBufferGL(OptiXLayer::ToneMappedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
		break;
	case 2:
		sutil::displayBufferGL(OptiXLayer::DenoisedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
		break;
	}

	{
		int k = OptiXLayer::Camera().StaticFrameNum();
		sutil::displayFps(k);
		static char text_[32];
		sprintf(text_, "frame: %d", k);
		string text = string(text_);
		sutil::displayText(text.c_str(), 200, 10);
	}
	//glutSwapBuffers();
}

void GLViewer::initializeGL()
{
	int a;
	char** b = NULL;
	glutInit(&a, b);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);

#ifndef __APPLE__
	glewInit();
#endif
	OptiXLayer::LazyInit();

	OptiXLayer::LoadScene();

	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	early_time = sutil::currentTime();
	std::thread([this] {
		while (true) {
			float now_time = sutil::currentTime();
			time -= now_time - early_time;
			early_time = now_time;
			if (pause) {
				Sleep(500);
				continue;
			}
			if (time < 0) {
				time = -1;
				QMetaObject::invokeMethod(this, "updateGL", Qt::ConnectionType(Qt::BlockingQueuedConnection | Qt::UniqueConnection));
			}
			else {
				Sleep(10);
			}
		}
	}).detach();
}

void GLViewer::resizeGL(int _w, int _h)
{
	if (OptiXLayer::ResizeBuffer(_w, _h)) {
		glViewport(0, 0, _w, _h);
		time = 1;
	}
}