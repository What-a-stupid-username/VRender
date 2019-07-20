#pragma once

#include "VRender.hpp"


namespace VRender {

	VRenderer& VRenderer::Instance() {
		static VRenderer& ins = VRenderer();
		return ins;
	}
}
