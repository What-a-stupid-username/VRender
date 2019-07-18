#pragma once

#include "PipelineUtility.hpp"

namespace VRender {
	optix::Context PipelineUtility::context;

	RTsize  PipelineUtility::target_width;
	RTsize	PipelineUtility::target_height;

	list<VDispatch> PipelineUtility::dispatchs;

	VCamera PipelineUtility::camera;
}