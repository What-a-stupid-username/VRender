#ifndef VECTOR_WIDGET_H
#define VECTOR_WIDGET_H

#include <QWidget>
#include <transform_widget.h>

namespace Ui {
class Vector_widget;
}

class Vector_widget : public QWidget
{
    Q_OBJECT

public:

    explicit Vector_widget(QWidget *parent = nullptr);
    ~Vector_widget();
public slots:

    void SetX(QString string);
    void SetY(QString string);
    void SetZ(QString string);

private:
	int index = -1;
	Transform_widget* parent;

	void SendChange(int, float);
};

#endif // VECTOR_WIDGET_H
