#include "OptiXLayer.h"


Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float3& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}
void setMaterial(
	GeometryInstance& gi,
	Material material,
	const std::string& color_name,
	const float& color)
{
	gi->addMaterial(material);
	gi[color_name]->setFloat(color);
}

GeometryInstance createParallelogram(Context context,
	const float3& anchor,
	const float3& offset1,
	const float3& offset2)
{
	Geometry parallelogram = context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setIntersectionProgram(pgram_intersection);
	parallelogram->setBoundingBoxProgram(pgram_bounding_box);

	float3 normal = normalize(cross(offset1, offset2));
	float d = dot(normal, anchor);
	float4 plane = make_float4(normal, d);

	float3 v1 = offset1 / dot(offset1, offset1);
	float3 v2 = offset2 / dot(offset2, offset2);

	parallelogram["plane"]->setFloat(plane);
	parallelogram["anchor"]->setFloat(anchor);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);

	GeometryInstance gi = context->createGeometryInstance();
	gi->setGeometry(parallelogram);
	return gi;
}

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
	camera_eye = make_float3(278.0f, 273.0f, -900.0f);
	camera_lookat = make_float3(278.0f, 273.0f, 0.0f);
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}

void OptiXLayer::Init() {
	try
	{
		context = Context::create();
		context->setRayTypeCount(3);
		context->setEntryPointCount(2);
		context->setStackSize(3600);
		//context->setMaxCallableProgramDepth(9);
		context->setMaxTraceDepth(9);

		context["scene_epsilon"]->setFloat(1.e-3f);
		context["common_ray_type"]->setUint(0u);
		context["pathtrace_shadow_ray_type"]->setUint(1u);
		context["photon_ray_type"]->setUint(2u);

		Buffer buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, screenWidth, screenHeight, false);
		context["output_buffer"]->set(buffer);

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
		const char *ptx = sutil::getPtxString("Renderer", "path_tracer.cu");
		context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "pathtrace_camera"));
		context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
		context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));

		ptx = sutil::getPtxString("Renderer", "photon_map.cu");
		context->setRayGenerationProgram(1, context->createProgramFromPTXString(ptx, "emmit"));
		context->setExceptionProgram(1, context->createProgramFromPTXString(ptx, "exception"));
		context->setMissProgram(1, context->createProgramFromPTXString(ptx, "miss"));

		context["sqrt_num_samples"]->setUint(sqrt_num_samples);
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

void OptiXLayer::LoadScene() {
	Instance().rendering.lock();

	try
	{
		auto context = Context();
		// Light buffer
		const float3 light_em = make_float3(15, 15, 15);
		ParallelogramLight light;
		light.corner = make_float3(343.0f, 548.6f, 227.0f);
		light.v1 = make_float3(0.0f, 0.0f, 105.0f);
		light.v2 = make_float3(-130.0f, 0.0f, 0.0f);
		light.normal = normalize(cross(light.v1, light.v2));
		light.emission = light_em;

		Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
		light_buffer->setFormat(RT_FORMAT_USER);
		light_buffer->setElementSize(sizeof(ParallelogramLight));
		light_buffer->setSize(1u);
		memcpy(light_buffer->map(), &light, sizeof(light));
		light_buffer->unmap();
		context["lights"]->setBuffer(light_buffer);


		// Set up material
		Material default_lit = context->createMaterial();
		{
			const char *ptx = sutil::getPtxString("Renderer", "path_tracer.cu");
			Program default_lit_closest_hit = context->createProgramFromPTXString(ptx, "default_lit_closest_hit");
			Program default_lit_any_hit = context->createProgramFromPTXString(ptx, "default_lit_any_hit");
			ptx = sutil::getPtxString("Renderer", "photon_map.cu");
			Program default_lit_photon_closest_hit = context->createProgramFromPTXString(ptx, "default_lit_photon_closest_hit");
			default_lit->setClosestHitProgram(0, default_lit_closest_hit);
			default_lit->setAnyHitProgram(1, default_lit_any_hit);
			default_lit->setClosestHitProgram(2, default_lit_photon_closest_hit);
		}


		Material default_light = context->createMaterial();
		{
			const char *ptx = sutil::getPtxString("Renderer", "path_tracer.cu");
			Program default_light_closest_hit = context->createProgramFromPTXString(ptx, "default_light_closest_hit");
			ptx = sutil::getPtxString("Renderer", "photon_map.cu");
			Program default_light_any_hit = context->createProgramFromPTXString(ptx, "light_ignore_photon_hit");
			default_light->setClosestHitProgram(0, default_light_closest_hit);
			default_light->setAnyHitProgram(2, default_light_any_hit);
		}

		{
			// Set up parallelogram programs
			const char *ptx = sutil::getPtxString("Renderer", "parallelogram.cu");
			pgram_bounding_box = context->createProgramFromPTXString(ptx, "bounds");
			pgram_intersection = context->createProgramFromPTXString(ptx, "intersect");
		}


		// create geometry instances
		std::vector<GeometryInstance> gis;

		const float3 white = make_float3(0.8f, 0.8f, 0.8f);
		const float3 blue = make_float3(0.05f, 0.05f, 0.8f);
		const float3 red = make_float3(0.8f, 0.05f, 0.05f);

		// Floor
		gis.push_back(createParallelogram(context, make_float3(0.0f, 0.0f, 0.0f),
			make_float3(0.0f, 0.0f, 559.2f),
			make_float3(556.0f, 0.0f, 0.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);

		// Ceiling
		gis.push_back(createParallelogram(context, make_float3(0.0f, 548.8f, 0.0f),
			make_float3(556.0f, 0.0f, 0.0f),
			make_float3(0.0f, 0.0f, 559.2f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "metallic", 1.f);

		// Back wall
		gis.push_back(createParallelogram(context, make_float3(0.0f, 0.0f, 559.2f),
			make_float3(0.0f, 548.8f, 0.0f),
			make_float3(556.0f, 0.0f, 0.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "smoothness", 1.f);

		// Right wall
		gis.push_back(createParallelogram(context, make_float3(0.0f, 0.0f, 0.0f),
			make_float3(0.0f, 548.8f, 0.0f),
			make_float3(0.0f, 0.0f, 559.2f)));
		setMaterial(gis.back(), default_lit, "albedo", blue);

		// Left wall
		gis.push_back(createParallelogram(context, make_float3(556.0f, 0.0f, 0.0f),
			make_float3(0.0f, 0.0f, 559.2f),
			make_float3(0.0f, 548.8f, 0.0f)));
		setMaterial(gis.back(), default_lit, "albedo", red);

		// Short block
		gis.push_back(createParallelogram(context, make_float3(130.0f, 165.0f, 65.0f),
			make_float3(-48.0f, 0.0f, 160.0f),
			make_float3(160.0f, 0.0f, 49.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "transparent", 1);
		gis.push_back(createParallelogram(context, make_float3(290.0f, 0.0f, 114.0f),
			make_float3(0.0f, 165.0f, 0.0f),
			make_float3(-50.0f, 0.0f, 158.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "transparent", 1);
		gis.push_back(createParallelogram(context, make_float3(130.0f, 0.0f, 65.0f),
			make_float3(0.0f, 165.0f, 0.0f),
			make_float3(160.0f, 0.0f, 49.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "transparent", 1);
		gis.push_back(createParallelogram(context, make_float3(82.0f, 0.0f, 225.0f),
			make_float3(0.0f, 165.0f, 0.0f),
			make_float3(48.0f, 0.0f, -160.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "transparent", 1);
		gis.push_back(createParallelogram(context, make_float3(240.0f, 0.0f, 272.0f),
			make_float3(0.0f, 165.0f, 0.0f),
			make_float3(-158.0f, 0.0f, -47.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "transparent", 1);

		// Tall block
		gis.push_back(createParallelogram(context, make_float3(423.0f, 330.0f, 247.0f),
			make_float3(-158.0f, 0.0f, 49.0f),
			make_float3(49.0f, 0.0f, 159.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "metallic", 1.f);
		setMaterial(gis.back(), default_lit, "smoothness", 1.f);
		gis.push_back(createParallelogram(context, make_float3(423.0f, 0.0f, 247.0f),
			make_float3(0.0f, 330.0f, 0.0f),
			make_float3(49.0f, 0.0f, 159.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "metallic", 1.f);
		setMaterial(gis.back(), default_lit, "smoothness", 1.f);
		gis.push_back(createParallelogram(context, make_float3(472.0f, 0.0f, 406.0f),
			make_float3(0.0f, 330.0f, 0.0f),
			make_float3(-158.0f, 0.0f, 50.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "metallic", 1.f);
		setMaterial(gis.back(), default_lit, "smoothness", 1.f);
		gis.push_back(createParallelogram(context, make_float3(314.0f, 0.0f, 456.0f),
			make_float3(0.0f, 330.0f, 0.0f),
			make_float3(-49.0f, 0.0f, -160.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "metallic", 1.f);
		setMaterial(gis.back(), default_lit, "smoothness", 1.f);
		gis.push_back(createParallelogram(context, make_float3(265.0f, 0.0f, 296.0f),
			make_float3(0.0f, 330.0f, 0.0f),
			make_float3(158.0f, 0.0f, -49.0f)));
		setMaterial(gis.back(), default_lit, "albedo", white);
		setMaterial(gis.back(), default_lit, "metallic", 1.f);
		setMaterial(gis.back(), default_lit, "smoothness", 1.f);

		// Create shadow group (no light)
		GeometryGroup shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
		shadow_group->setAcceleration(context->createAcceleration("Trbvh"));
		context["top_shadower"]->set(shadow_group);

		// Light
		gis.push_back(createParallelogram(context, make_float3(343.0f, 548.6f, 227.0f),
			make_float3(0.0f, 0.0f, 105.0f),
			make_float3(-130.0f, 0.0f, 0.0f)));
		setMaterial(gis.back(), default_light, "emission_color", light_em);

		// Create geometry group
		GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
		geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
		context["top_object"]->set(geometry_group);

		Instance().dirty = true;
	}
	catch (const std::exception&)
	{

		Instance().rendering.unlock();
		return;
	}	
	Instance().rendering.unlock();
}

void OptiXLayer::RenderResult(uint maxFrame) {
	auto& layer = Instance();
	if (layer.pause) return;
	layer.rendering.lock();
	layer.camera.UpdateOptiXContext(layer.context, layer.screenWidth, layer.screenHeight, layer.dirty);
	if (layer.camera.staticFrameNum > maxFrame) return;
	try {
		layer.context->validate();
		layer.cb->validate();

		layer.context["rnd_seed"]->setUint(rand() % 999);
		Variable(layer.tonemapStage->queryVariable("exposure"))->setFloat(layer.exposure);

		auto length_buffer = layer.context["length_buffer"]->getBuffer();
		auto length = (int*)length_buffer->map();
		length[0] = 0; length[1] = 0; length[2] = 0;
		length_buffer->unmap();

		layer.dirty = false;
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

	sutil::displayBufferPPM(&name[0], layer.GetResult());

	layer.rendering.unlock();
}
