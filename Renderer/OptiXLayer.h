#pragma once

#include "CommonInclude.h"

class Camera {
	friend class OptiXLayer;
private:
	// Camera state
	float3         camera_up;
	float3         camera_lookat;
	float3         camera_eye;
	Matrix4x4      camera_rotate;

	uint staticFrameNum = 1;
	bool dirty = true;

	bool UpdateOptiXContext(Context context, float width, float height, bool fource = false);
public:
	Camera();
	
	inline void Scale(float scale) {
		dirty = true;
		camera_eye = camera_eye + (camera_lookat - camera_eye) * scale;
	}
	inline void Rotate(Matrix4x4 rotate) {
		dirty = true;
		camera_rotate = rotate;
	}
	inline const uint StaticFrameNum() {
		return staticFrameNum;
	}
};


class OptiXLayer {
//properties
private:
	Context        context = 0;

	uint32_t       screenWidth = 512;
	uint32_t       screenHeight = 512;

	int            sqrt_num_samples = 1;
	int            rr_begin_depth = 1;

	bool dirty = true;

	Camera camera;

	PostprocessingStage tonemapStage, denoiserStage;

	float exposure = 3.f, gamma = 2.2f;

	CommandList cb;
private:
	void Init();
	OptiXLayer() {
		//Init();
	}


	OptiXLayer(const OptiXLayer&) abandon;
	OptiXLayer& operator=(const OptiXLayer&) abandon;
public:
	static OptiXLayer& Instance();
	static void LazyInit() {
		Instance().Init();
	}

	inline static Buffer OriginBuffer() {
		return Instance().context["output_buffer"]->getBuffer();
	}

	inline static Buffer ToneMappedBuffer() {
		return Instance().context["tonemapped_buffer"]->getBuffer();
	}

	inline static Buffer DenoisedBuffer() {
		return Instance().context["output_denoisedBuffer"]->getBuffer();
	}

	inline const static Context Context() {
		return Instance().context;
	}

	static void Release() {
		auto& layer = Instance();
		if (layer.context) {
			layer.context->destroy();
		}
		layer.context = 0;
	}

	static void ClearScene() {
		Release();
		Instance().Init();
	}

	static void LoadScene();

	inline static Camera& Camera() {
		return Instance().camera;
	}

	inline static void ChangeExposure(float value) {
		Instance().exposure += value;
	}

	static void RenderResult(uint maxFrame = 9999);

	static bool ResizeBuffer(int& w, int& h);

	static void RebuildCommandList(bool openPost = false, bool remain = true);

	inline static void GetBufferSize(float& w, float& h) {
		auto& layer = Instance();
		w = layer.screenWidth;
		h = layer.screenHeight;
	}
};
