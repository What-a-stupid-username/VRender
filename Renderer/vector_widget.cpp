#include "vector_widget.h"
#include <iostream>
using namespace std;

Vector_widget::Vector_widget(QWidget *parent) :
    QWidget(parent)
{
	this->parent = (Transform_widget*)parent;
}

Vector_widget::~Vector_widget()
{
}

void Vector_widget::SetX(QString str)
{
	bool vali = false; 
	float v = str.toFloat(&vali);
	if (vali) {
		SendChange(0, v);
	}
}
void Vector_widget::SetY(QString string)
{
	bool vali = false;
	float v = string.toFloat(&vali);
	if (vali) {
		SendChange(1, v);
	}
}
void Vector_widget::SetZ(QString string)
{
	bool vali = false;
	float v = string.toFloat(&vali);
	if (vali) {
		SendChange(2, v);
	}
}

void Vector_widget::SendChange(int cmp, float v) {
	if (index == -1) {
		string name = this->objectName().toStdString();
		index = name[name.length() - 1] - '0';
	}
	parent->ChangeTransform(index, cmp, v);
}