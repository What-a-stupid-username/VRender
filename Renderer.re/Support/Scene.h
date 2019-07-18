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
		if (root != NULL) {
			root->Release();
		}

		auto& layer = OptiXLayer::Instance();


		try
		{
			auto context = layer.Context();

			// Light buffer
			const float3 light_em = make_float3(15, 15, 15);
			ParallelogramLight light;
			light.corner = make_float3(1, 5, 1);
			light.v1 = make_float3(0, 0, 2);
			light.v2 = make_float3(-2, 0, 0);
			light.normal = normalize(cross(light.v1, light.v2));
			light.emission = light_em;

			Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
			light_buffer->setFormat(RT_FORMAT_USER);
			light_buffer->setElementSize(sizeof(ParallelogramLight));
			light_buffer->setSize(1u);
			memcpy(light_buffer->map(), &light, sizeof(light));
			light_buffer->unmap();
			context["lights"]->setBuffer(light_buffer);

			const float3 white = make_float3(0.8f, 0.8f, 0.8f);
			const float3 blue = make_float3(0.05f, 0.05f, 0.8f);
			const float3 red = make_float3(0.8f, 0.05f, 0.05f);

			class Foo {
			public:
				static VObject* foo(float3 position, float3 rotation, float3 scale, string material = "default") {
					float3 anchor = make_float3(-5, 0, -5);
					float3 offset1 = make_float3(0, 0, 10);
					float3 offset2 = make_float3(10, 0, 0);

					VObject * obj = new VObject("parallelogram");

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

					*obj->Transform()->Position<float3>() = position;
					*obj->Transform()->Rotation<float3>() = rotation;
					*obj->Transform()->Scale<float3>() = scale;
					return obj;
				}
				static VObject* foo2(float3 position, float3 rotation, float3 scale, string material = "default") {
					VObject * obj = new VObject("triangle_mesh");

					obj->GeometryFilter()->SetMesh(VMesh::Find("Cube.obj"));

					obj->SetMaterial(VMaterial::Find(material));

					*obj->Transform()->Position<float3>() = position;
					*obj->Transform()->Rotation<float3>() = rotation;
					*obj->Transform()->Scale<float3>() = scale;
					return obj;
				}
			};
			// Floor
			Foo::foo(make_float3(0, -5, 0), make_float3(0, 0, 0), make_float3(1, 1, 1));
			// Ceiling
			Foo::foo(make_float3(0, 5, 0), make_float3(-180, 0, 0), make_float3(1, 1, 1));
			// Back wall
			Foo::foo(make_float3(0, 0, 5), make_float3(-90, 0, 0), make_float3(1, 1, 1), "default_feb");
			// Right wall
			Foo::foo(make_float3(-5, 0, 0), make_float3(0, 0, -90), make_float3(1, 1, 1), "default_blue");
			//Left wall
			Foo::foo(make_float3(5, 0, 0), make_float3(0, 0, 90), make_float3(1, 1, 1), "default_red");

			//Light
			Foo::foo(make_float3(0, 4.999, 0), make_float3(-180, 0, 0), make_float3(0.2, 0.2, 0.2), "light");

			Foo::foo2(make_float3(0, 1, 0), make_float3(0, 0, 0), make_float3(1, 1, 1), "default_red");

			root = VTransform::Root();
			cout << root->group->getChildCount() << endl;
		}
		catch (const Exception & e)
		{
			cout << e.getErrorString() << endl;
		}
	}

public:
	static VTransform*  root;
public:

	static void LoadScene(string path) {
		OptiXLayer::LoadScene(LoadFromFile);
	}
};