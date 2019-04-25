#pragma once

#include "CommonInclude.h"
#include "Components.h"

class OptiXLayer;
class VScene {
	friend class OptiXLayer;
private:
	static void LoadCornell();

	static string scene_path;

	static void LoadFromFile() {
		//todo:

	}

public:
	static VTransform*  root;
public:

	static void LoadScene(string path) {
		OptiXLayer::LoadScene(LoadFromFile);
	}
};