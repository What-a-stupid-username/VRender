#ifndef Transform_widget_H
#define Transform_widget_H

#include <QWidget>

namespace Ui {
class Transform_widget;
}

class Transform_widget : public QWidget
{
    Q_OBJECT

public:
	float x, y, z;

    explicit Transform_widget(QWidget *parent = nullptr);
    ~Transform_widget();

	void ChangeTransform(int index, int cmp, float v);

private:
    Ui::Transform_widget *ui;
};

#endif // Transform_widget_H
