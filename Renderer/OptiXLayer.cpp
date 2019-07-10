#include "OptiXLayer.h"
#include "Support/Components.h"
#include "Support/Scene.h"

bool Camera::UpdateOptiXContext(Context context, float width, float height, bool fource) {
	try
	{
		if (fource) dirty = true;
		if (!dirty) {
			context["frame_number"]->setUint(staticFrameNum++);
			return false;
		}
		dirty = false;
		const float fov = 35.0f;
		const float aspect_ratio = width / height;

		float3 camera_u, camera_v, camera_w;
		sutil::calculateCameraVariables(
			camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
			camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

		const Matrix4x4 frame = Matrix4x4::fromBasis(
			normalize(camera_u),
			normalize(camera_v),
			normalize(-camera_w),
			camera_lookat);
		const Matrix4x4 frame_inv = frame.inverse();
		// Apply camera rotation twice to match old SDK behavior
		const Matrix4x4 trans = frame * camera_rotate*camera_rotate*frame_inv;

		camera_eye = make_float3(trans*make_float4(camera_eye, 1.0f));
		camera_lookat = make_float3(trans*make_float4(camera_lookat, 1.0f));
		camera_up = make_float3(trans*make_float4(camera_up, 0.0f));

		sutil::calculateCameraVariables(
			camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
			camera_u, camera_v, camera_w, true);

		camera_rotate = Matrix4x4::identity();

		staticFrameNum = 1;
		context["frame_number"]->setUint(staticFrameNum);
		context["eye"]->setFloat(camera_eye);
		context["U"]->setFloat(camera_u);
		context["V"]->setFloat(camera_v);
		context["W"]->setFloat(camera_w);
		//if (camera_changed) // reset accumulation
		//	frame_number = 1;
		//camera_changed = false;

		//context["frame_number"]->setUint(frame_number++);
		//context["eye"]->setFloat(camera_eye);
		//context["U"]->setFloat(camera_u);
		//context["V"]->setFloat(camera_v);
		//context["W"]->setFloat(camera_w);
	}
	catch (const std::exception&)
	{
		return false;
	}
	
	return true;
}

Camera::Camera() {
	camera_eye = make_float3(0, 0, -21);
	camera_lookat = make_float3(0, 0, 0);
	camera_up = make_float3(0, 1, 0);

	camera_rotate = Matrix4x4::identity();
}

void OptiXLayer::Init() {
	try
	{
#ifdef FORCE_NOT_USE_RTX
		int not_use_rtx = 0;
		rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(int), (void*)& not_use_rtx);
#endif // FOURCE_NOT_USE_RTX


		context = Context::create();
		context->setRayTypeCount(3);
		context->setEntryPointCount(2);

		context->setStackSize(2000);


		context["scene_epsilon"]->setFloat(1.e-3f);
		context["common_ray_type"]->setUint(0u);
		context["pathtrace_shadow_ray_type"]->setUint(1u);
		context["photon_ray_type"]->setUint(2u);

		Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, screenWidth, screenHeight, false);
		context["output_buffer"]->set(buffer);
		Buffer helperBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, screenWidth, screenHeight, false);
		context["helper_buffer"]->set(helperBuffer);

		Buffer denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, screenWidth, screenHeight, false);
		context["output_denoisedBuffer"]->set(denoisedBuffer);

		Buffer tonemappedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, screenWidth, screenHeight, false);
		context["tonemapped_buffer"]->set(tonemappedBuffer);

		Buffer length_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_INT, 3);
		context["length_buffer"]->set(length_buffer);

		Buffer photon_buffer;
		photon_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, 10000000);
		photon_buffer->setElementSize(sizeof(Photon));
		context["photon_buffer"]->set(photon_buffer);

		//Buffer photon_kdtree_buffer;
		//photon_kdtree_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT2, 10000000);
		//context["photon_kdtree_buffer"]->set(photon_kdtree_buffer);

		//Buffer photon_sorted_pointer_buffer;
		//photon_sorted_pointer_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT, 3, 10000000);
		//context["photon_sorted_pointer_buffer"]->set(photon_sorted_pointer_buffer);

		// Setup programs
		const char *ptx = sutil::getPtxString("Shaders", "path_tracer_camera.cu");
		context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "path_tracer_camera"));
		context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
		context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));

		ptx = sutil::getPtxString("Shaders", "photon_map.cu");
		context->setRayGenerationProgram(1, context->createProgramFromPTXString(ptx, "emmit"));
		context->setExceptionProgram(1, context->createProgramFromPTXString(ptx, "exception"));
		context->setMissProgram(1, context->createProgramFromPTXString(ptx, "miss"));

		context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
		context["bg_color"]->setFloat(make_float3(0.0f));

		tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");

		tonemapStage->declareVariable("input_buffer")->set(buffer);
		tonemapStage->declareVariable("output_buffer")->set(tonemappedBuffer);
		tonemapStage->declareVariable("exposure")->setFloat(exposure);
		tonemapStage->declareVariable("gamma")->setFloat(gamma);

		denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");

		denoiserStage->declareVariable("input_buffer")->set(tonemappedBuffer);
		denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
		denoiserStage->declareVariable("blend")->setFloat(0);

		cb = context->createCommandList();
#ifndef QT
		RebuildCommandList();
#endif // !QT

		dirty = true;
	}
	catch (Exception& e)
	{
		cout << "Init failed" << endl;
		cout << e.getErrorString() << endl;
	}
	
}

OptiXLayer & OptiXLayer::Instance() {
	static OptiXLayer& layer = OptiXLayer();
	return layer;
}

void OptiXLayer::LoadScene(function<void()> func) {
	auto& layer = Instance();
	layer.rendering.lock();

	if (func == nullptr) {
		func = layer.scene_func;
	}
	else {
		layer.scene_func = func;
	}
	if (func == nullptr) {
		func = VScene::LoadCornell;
		layer.scene_func = func;
	}

	func();

	auto& context = Context();

	VScene::root->group->validate();

	context["top_shadower"]->set(VScene::root->group);

	context["top_object"]->set(VScene::root->group);

	layer.MarkDirty();

	layer.rendering.unlock();
}

void OptiXLayer::RenderResult(uint maxFrame) {
	auto& layer = Instance();
	if (layer.pause) return;
	if (layer.camera.staticFrameNum > maxFrame) return;

	layer.rendering.lock();
	layer.camera.UpdateOptiXContext(layer.context, layer.screenWidth, layer.screenHeight, layer.dirty);
	try {
		layer.context->validate();
		layer.cb->validate();

		layer.context["rnd_seed"]->setUint(rand());
		layer.context["diffuse_strength"]->setFloat(layer.diffuse_strength);
		layer.context["max_depth"]->setInt(layer.max_depth);
#ifdef OPTIX_6
		layer.context->setMaxTraceDepth(layer.max_depth + 2);
#endif // OPTIX_6
		layer.context["cut_off_high_variance_result"]->setUint(layer.cut_off_high_variance_result);
		layer.context["sqrt_num_samples"]->setUint(layer.sqrt_num_samples);
		Variable(layer.tonemapStage->queryVariable("exposure"))->setFloat(layer.exposure);

		//auto length_buffer = layer.context["length_buffer"]->getBuffer();
		//auto length = (int*)length_buffer->map();
		//length[0] = 0; length[1] = 0; length[2] = 0;
		//length_buffer->unmap();

		layer.dirty = false;

		if (VMaterial::ApplyAllChanges()) layer.dirty = true;
		if (VTransform::ApplyAllChanges()) {
			layer.dirty = true;
			VScene::root->group->getAcceleration()->markDirty();
		}
		layer.cb->execute();
	}
	catch (Exception& e) {
		cout << e.getErrorString() << endl;
		layer.rendering.unlock();
		return;
	}
	layer.rendering.unlock();
}

bool OptiXLayer::ResizeBuffer(int & w, int & h) {
	sutil::ensureMinimumSize(w, h);

	auto& layer = Instance();
	if (w == layer.screenWidth&& h == layer.screenHeight) return false;

	layer.rendering.lock();
	layer.screenWidth = w;
	layer.screenHeight = h;

	sutil::resizeBuffer(OptiXLayer::OriginBuffer(), w, h);
	sutil::resizeBuffer(OptiXLayer::HelperBuffer(), w, h);
	sutil::resizeBuffer(OptiXLayer::ToneMappedBuffer(), w, h);
	sutil::resizeBuffer(OptiXLayer::DenoisedBuffer(), w, h);

	layer.rendering.unlock();
	RebuildCommandList();

	layer.rendering.lock();
	layer.camera.UpdateOptiXContext(layer.context, layer.screenWidth, layer.screenHeight, true);
	layer.rendering.unlock();

	return true;
}

void OptiXLayer::RebuildCommandList(bool openPost) {
	auto& layer = Instance();
	layer.rendering.lock();
	layer.cb->destroy();
	layer.cb = layer.context->createCommandList();
	layer.cb->appendLaunch(0, layer.screenWidth, layer.screenHeight);
	//layer.cb->appendLaunch(1, 1000000, 1);
	if (openPost) {
		layer.cb->appendPostprocessingStage(layer.tonemapStage, layer.screenWidth, layer.screenHeight);
		layer.cb->appendPostprocessingStage(layer.denoiserStage, layer.screenWidth, layer.screenHeight);
	}
	layer.cb->finalize();
	layer.rendering.unlock();
}

void OptiXLayer::SaveResultToFile(string name)
{
	auto& layer = Instance();

	layer.rendering.lock();

	sutil::displayBufferBMP(&name[0], layer.GetResult(), false);

	layer.rendering.unlock();
}
