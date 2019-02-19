#include "transform_widget.h"
#include <iostream>
using namespace std;

Transform_widget::Transform_widget(QWidget *parent) :
    QWidget(parent)
{
}

Transform_widget::~Transform_widget()
{
}

void Transform_widget::ChangeTransform(int index, int cmp, float v) {
	cout << index << ", " << cmp << ": " << v << endl;
}