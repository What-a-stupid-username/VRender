#ifndef GLVIEWER_H
#define GLVIEWER_H

#include "CommonInclude.h"
#include "OptiXLayer.h"

#include <QtOpenGL>

#include <QColorDialog>


class GLViewer : public QGLWidget
{
	Q_OBJECT

public:
	bool pause = false;

	GLViewer(QWidget *parent = 0);
	~GLViewer();

	public slots:
	
	void setBackgroundColor() {
		QColor color = QColorDialog::getColor(Qt::black, this, tr("Set Background Color!"));
		if (!color.isValid()) return;
		this->qglClearColor(color);
		this->updateGL();
	}

	void Pause() {
		pause = !pause;

	}

	void PostProcess() {
		static bool on = false;
		on = !on;
		OptiXLayer::RebuildCommandList(on, false);
		if (on) buffer = 2;
		else buffer = 0;
	}

protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void wheelEvent(QWheelEvent *event);

	void initializeGL();
	void resizeGL(int _w, int _h);
	void paintGL();

private:
	// Mouse state
	int2           mouse_prev_pos;
	int            mouse_button;
	//helper
	sutil::Arcball arcball;
	float time = 0;
	float early_time = 0;

	int buffer = 0;
};

#endif // GLVIEWER_H
