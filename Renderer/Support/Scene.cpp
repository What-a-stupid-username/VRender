#include "Components.h"
#include "Scene.h"

VTransform* VScene::root;
string VScene::scene_path;

void VScene::LoadCornell() {

	auto& layer = OptiXLayer::Instance();

	try
	{
		auto context = layer.Context();
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


		// create geometry instances
		std::vector<GeometryInstance> gis;

		const float3 white = make_float3(0.8f, 0.8f, 0.8f);
		const float3 blue = make_float3(0.05f, 0.05f, 0.8f);
		const float3 red = make_float3(0.8f, 0.05f, 0.05f);

		class Foo {
		public:
			static VObject* foo(float3& anchor, float3& offset1, float3& offset2, string material = "default") {
				VObject* obj = new VObject("parallelogram");

				float3 normal = normalize(cross(offset1, offset2));
				float d = dot(normal, anchor);
				float4 plane = make_float4(normal, d);

				float3 v1 = offset1 / dot(offset1, offset1);
				float3 v2 = offset2 / dot(offset2, offset2);

				obj->GeometryFilter()->Visit("plane")->setFloat(plane);
				obj->GeometryFilter()->Visit("anchor")->setFloat(anchor);
				obj->GeometryFilter()->Visit("v1")->setFloat(v1);
				obj->GeometryFilter()->Visit("v2")->setFloat(v2);

				obj->SetMaterial(VMaterial::Find(material));
				return obj;
			}
		};
		// Floor
		Foo::foo(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 559.2f), make_float3(556.0f, 0.0f, 0.0f));
		// Ceiling
		Foo::foo(make_float3(0.0f, 548.8f, 0.0f), make_float3(556.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 559.2f));
		// Back wall
		Foo::foo(make_float3(0.0f, 0.0f, 559.2f), make_float3(0.0f, 548.8f, 0.0f), make_float3(556.0f, 0.0f, 0.0f), "default_feb");
		// Right wall
		Foo::foo(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 548.8f, 0.0f), make_float3(0.0f, 0.0f, 559.2f), "default_blue");
		// Left wall
		Foo::foo(make_float3(556.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 559.2f), make_float3(0.0f, 548.8f, 0.0f), "default_red");
		// Short block
		Foo::foo(make_float3(130.0f, 165.0f, 65.0f), make_float3(-48.0f, 0.0f, 160.0f), make_float3(160.0f, 0.0f, 49.0f), "default_transparent");
		Foo::foo(make_float3(290.0f, 0.0f, 114.0f), make_float3(0.0f, 165.0f, 0.0f), make_float3(-50.0f, 0.0f, 158.0f), "default_transparent");
		Foo::foo(make_float3(130.0f, 0.0f, 65.0f), make_float3(0.0f, 165.0f, 0.0f), make_float3(160.0f, 0.0f, 49.0f), "default_transparent");
		Foo::foo(make_float3(82.0f, 0.0f, 225.0f), make_float3(0.0f, 165.0f, 0.0f), make_float3(48.0f, 0.0f, -160.0f), "default_transparent");
		Foo::foo(make_float3(240.0f, 0.0f, 272.0f), make_float3(0.0f, 165.0f, 0.0f), make_float3(-158.0f, 0.0f, -47.0f), "default_transparent");

		//// Tall block
		Foo::foo(make_float3(423.0f, 330.0f, 247.0f), make_float3(-158.0f, 0.0f, 49.0f), make_float3(49.0f, 0.0f, 159.0f), "default_mirror");
		Foo::foo(make_float3(423.0f, 0.0f, 247.0f), make_float3(0.0f, 330.0f, 0.0f), make_float3(49.0f, 0.0f, 159.0f), "default_mirror");
		Foo::foo(make_float3(472.0f, 0.0f, 406.0f), make_float3(0.0f, 330.0f, 0.0f), make_float3(-158.0f, 0.0f, 50.0f), "default_mirror");
		Foo::foo(make_float3(314.0f, 0.0f, 456.0f), make_float3(0.0f, 330.0f, 0.0f), make_float3(-49.0f, 0.0f, -160.0f), "default_mirror");
		Foo::foo(make_float3(265.0f, 0.0f, 296.0f), make_float3(0.0f, 330.0f, 0.0f), make_float3(158.0f, 0.0f, -49.0f), "default_mirror");

		//Light
		Foo::foo(make_float3(343.0f, 548.6f, 227.0f), make_float3(0.0f, 0.0f, 105.0f), make_float3(-130.0f, 0.0f, 0.0f), "light");

		//VObject* obj = new VObject("triangle_mesh");
		//VMesh* mesh = VMesh::Find("Test1.OBJ");
		//obj->GeometryFilter()->SetMesh(mesh);
		//obj->SetMaterial(VMaterial::Find("default"));


		//float3 center = make_float3(240.0f, 240.0f, 240.0f);
		//float3 scale = make_float3(2, 2, 2);
		//float3 rota = make_float3(0, 90, 0);
		//*obj->Transform()->Position<float3>() = center;
		//*obj->Transform()->Scale<float3>() = scale;
		//*obj->Transform()->Rotation<float3>() = rota;

		root = VTransform::Root();
	}
	catch (const Exception& e)
	{
		cout << e.getErrorString() << endl;
	}
}
