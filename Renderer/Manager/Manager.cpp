#include "Manager.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../Support/stb_image.h"

#include "../Support/objLoader.h"

namespace VRender {

	namespace prime {

		void VShaderObj::LoadFromFile(const string& shader_name) {
			try {
				closestHitPrograms.clear();
				anyHitPrograms.clear();
				auto context = OptixInstance::Instance().Context();
				string all_contain;
				std::ifstream file((std::string(sutil::samplesDir()) + "/Shaders/" + shader_name + ".cu").c_str());
				if (file.good()) {
					// Found usable source file
					std::stringstream source_buffer;
					source_buffer << file.rdbuf();
					all_contain = source_buffer.str();
				}
				const char* ptx = sutil::getPtxString("Shaders", (shader_name + ".cu").c_str());

				regex r("#pragma (.*?) (.*?)\n");
				sregex_iterator it(all_contain.begin(), all_contain.end(), r);
				sregex_iterator end;
				for (; it != end; ++it)
				{
					int ray_type = Shit<int>::ToProperty(it->operator[](1));
					string program_type = Shit<string>::ToProperty(it->operator[](2));
					if (program_type == "ClosestHit") {
						Program program = context->createProgramFromPTXString(ptx, shader_name + "_ClosestHit");
						closestHitPrograms[ray_type] = VProgram(new ProgramWrapper(program));
					}
					else if (program_type == "AnyHit") {
						Program program = context->createProgramFromPTXString(ptx, shader_name + "_AnyHit");
						anyHitPrograms[ray_type] = VProgram(new ProgramWrapper(program));
					}
				}
			}
			catch (Exception & e) {
				cout << e.getErrorString() << endl;
				system("PAUSE");
			}
		}

		void VShaderObj::ApplyShader(optix::Material mat) {
			for each (auto pair in closestHitPrograms)
				mat->setClosestHitProgram(pair.first, pair.second->program);
			for each (auto pair in anyHitPrograms)
				mat->setAnyHitProgram(pair.first, pair.second->program);
		}

		VTextureObj::VTextureObj(const string & path) {
			auto& context = OptixInstance::Instance().Context();
			if (path != "")
			{
				buffer = context->createBuffer(RT_BUFFER_INPUT);
				int w, h, n;
				auto img = stbi_load((string(sutil::projectDir()) + "/Textures/" + path).c_str(), &w, &h, &n, STBI_rgb_alpha);

				n = 4;
				buffer->setFormat(RT_FORMAT_UNSIGNED_BYTE4);

				buffer->setSize(w, h);

				char* ptr = (char*)buffer->map();

				memcpy(ptr, img, sizeof(unsigned char) * n * w * h);

				delete(img);
				buffer->unmap();

				sampler = context->createTextureSampler();
				sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
				sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
				sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
				regex r(".sRGB");
				sregex_iterator it(path.begin(), path.end(), r);
				sregex_iterator end;
				bool sRGB = (it != end);
				if (sRGB)
					sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB);
				else
					sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
				sampler->setMaxAnisotropy(1);
				sampler->setMipLevelClamp(0, 1);
				sampler->setArraySize(1);
				sampler->setBuffer(buffer);
				id = sampler->getId();
			}
		}

		VTextureObj::~VTextureObj() {
			SAFE_RELEASE_OPTIX_OBJ(buffer);
			SAFE_RELEASE_OPTIX_OBJ(sampler);
			id = 0;
		}

		void VMaterialObj::Load(const string& name) {
			try
			{
				textures.clear();
				for each (auto pair in properties)
					pair.second.Release();
				properties.clear();
				shader = nullptr;

				this->name = name;

				if (name == "error") throw exception();

				PropertyReader reader("/Materials", name + ".txt");
				properties = reader.GetAllProperties();

				string shader_name = reader.GetPropertyValue<string>("Shader");
				shader = VShaderManager::Find(shader_name);

				for each (auto pair in properties)
				{
					if (pair.second.Type() == "string") {
						int k1 = pair.first.find('|');
						if (k1 != -1) {
							string special_type = pair.first.substr(0, k1);
							string name = pair.first.substr(k1 + 1, pair.first.length() - k1 - 1);
							if (special_type == "Texture") {
								auto tex_name = *pair.second.GetData<string>();
								auto tex = VTextureManager::Find(tex_name);
								textures[tex_name] = tex;
							}
						}
					}
				}
			}
			catch (const std::exception&)
			{
				string shader_name = "error";
				shader = VShaderManager::Find(shader_name);
			}
		}

		VMeshObj::VMeshObj(const string & name) {
			int k = name.find_last_of('.');
			if (k == -1) throw Exception("ERROR mesh name.");

			this->name = name;

			string format = name.substr(k, name.length() - k);

			vector<float3> verts;
			vector<float3> normals;
			vector<float2> texcoords;
			vector<int3> v_index;
			vector<int3> t_index;
			vector<int3> n_index;

			if (format == ".obj" || format == ".OBJ") {
				tinyobj::attrib_t attrib;
				std::vector<tinyobj::shape_t> shapes;
				string warn, err;
				tinyobj::LoadObj(&attrib, &shapes, NULL, &warn, &err, (string(sutil::projectDir()) + "/Meshs/" + name).c_str());
				if (err.length() != 0) {
					err = "";
					tinyobj::LoadObj(&attrib, &shapes, NULL, &warn, &err, (string(sutil::samplesDir()) + "/Meshs/" + name).c_str());
				}
				if (err.length() != 0)
					throw Exception("ERROR read mesh.\n" + err);
				if (warn.length() != 0) cout << warn << endl;

				for (int i = 0; i < attrib.vertices.size() / 3; i++)
				{
					float3 v;
					v.x = attrib.vertices[i * 3];
					v.y = attrib.vertices[i * 3 + 1];
					v.z = attrib.vertices[i * 3 + 2];
					verts.push_back(v);
				}
				for (int i = 0; i < attrib.normals.size() / 3; i++)
				{
					float3 v;
					v.x = attrib.normals[i * 3];
					v.y = attrib.normals[i * 3 + 1];
					v.z = attrib.normals[i * 3 + 2];
					normals.push_back(v);
				}
				for (int i = 0; i < attrib.texcoords.size() / 2; i++)
				{
					float2 v;
					v.x = attrib.texcoords[i * 2];
					v.y = attrib.texcoords[i * 2 + 1];
					texcoords.push_back(v);
				}
				for (int j = 0; j < shapes.size(); j++)
				{
					for (int i = 0; i < shapes[j].mesh.indices.size() / 3; i++)
					{
						int3 v;
						v.x = shapes[j].mesh.indices[i * 3].vertex_index;
						v.y = shapes[j].mesh.indices[i * 3 + 1].vertex_index;
						v.z = shapes[j].mesh.indices[i * 3 + 2].vertex_index;
						v_index.push_back(v);
					}
					for (int i = 0; i < shapes[j].mesh.indices.size() / 3; i++)
					{
						int3 v;
						v.x = shapes[j].mesh.indices[i * 3].texcoord_index;
						v.y = shapes[j].mesh.indices[i * 3 + 1].texcoord_index;
						v.z = shapes[j].mesh.indices[i * 3 + 2].texcoord_index;
						t_index.push_back(v);
					}
					for (int i = 0; i < shapes[j].mesh.indices.size() / 3; i++)
					{
						int3 v;
						v.x = shapes[j].mesh.indices[i * 3].normal_index;
						v.y = shapes[j].mesh.indices[i * 3 + 1].normal_index;
						v.z = shapes[j].mesh.indices[i * 3 + 2].normal_index;
						n_index.push_back(v);
					}
				}
			}
			else
			{
				throw Exception("ERROR unsupported mesh file format.");
			}


			auto& context = OptixInstance::Instance().Context();

			{//vert
				{
					vert_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, verts.size());
					auto ptr = vert_buffer->map();
					memcpy(ptr, verts.data(), verts.size() * sizeof(float3));
					vert_buffer->unmap();
				}
				{
					v_index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, v_index.size());
					auto ptr = v_index_buffer->map();
					memcpy(ptr, v_index.data(), v_index.size() * sizeof(int3));
					v_index_buffer->unmap();
				}
			}
			if (!normals.empty())//normal
			{
				{
					normal_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, normals.size());
					auto ptr = normal_buffer->map();
					memcpy(ptr, normals.data(), normals.size() * sizeof(float3));
					normal_buffer->unmap();
				}
				{
					n_index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, n_index.size());
					auto ptr = n_index_buffer->map();
					memcpy(ptr, n_index.data(), n_index.size() * sizeof(float3));
					n_index_buffer->unmap();
				}
			}
			if (!texcoords.empty())//texcoord
			{
				{
					tex_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, texcoords.size());
					auto ptr = tex_buffer->map();
					memcpy(ptr, texcoords.data(), texcoords.size() * sizeof(float2));
					tex_buffer->unmap();
				}
				{
					t_index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, t_index.size());
					auto ptr = t_index_buffer->map();
					memcpy(ptr, t_index.data(), t_index.size() * sizeof(int3));
					t_index_buffer->unmap();
				}
			}
		}

		VMeshObj::~VMeshObj() {
			SAFE_RELEASE_OPTIX_OBJ(vert_buffer);
			SAFE_RELEASE_OPTIX_OBJ(normal_buffer);
			SAFE_RELEASE_OPTIX_OBJ(tex_buffer);
			SAFE_RELEASE_OPTIX_OBJ(v_index_buffer);
			SAFE_RELEASE_OPTIX_OBJ(n_index_buffer);
			SAFE_RELEASE_OPTIX_OBJ(t_index_buffer);
		}
	}



	unordered_map<string, weak_ptr<prime::VShaderObj>> VShaderManager::shader_cache;

	VShader VShaderManager::Find(const string& name) {
		do
		{
			auto pair = shader_cache.find(name);
			if (pair != shader_cache.end()) {
				if (pair->second.expired()) {
					shader_cache.erase(name);
					break;
				}
				else
				{
					return pair->second.lock();
				}
			}
		} while (false);

		auto shader = VShader(new prime::VShaderObj(name));
		shader_cache[name] = shader;
		return shader;
	}



	unordered_map<string, weak_ptr<prime::VTextureObj>> VTextureManager::texture_cache;

	VTexture VTextureManager::Find(const string& name) {
		do
		{
			auto pair = texture_cache.find(name);
			if (pair != texture_cache.end()) {
				if (pair->second.expired()) {
					texture_cache.erase(name);
					break;
				}
				else
				{
					return pair->second.lock();
				}
			}
		} while (false);

		auto texture = VTexture(new prime::VTextureObj(name));
		texture_cache[name] = texture;
		return texture;
	}



	unordered_map<string, weak_ptr<prime::VMaterialObj>> VMaterialManager::material_cache;
	unordered_set<string> VMaterialManager::dirty_maretial;

	VMaterial VMaterialManager::Find(const string& name) {
		do
		{
			auto pair = material_cache.find(name);
			if (pair != material_cache.end()) {
				if (pair->second.expired()) {
					material_cache.erase(name);
					break;
				}
				else
				{
					return pair->second.lock();
				}
			}
		} while (false);

		auto mat = VMaterial(new prime::VMaterialObj(name));
		material_cache[name] = mat;
		MarkDirty(mat);
		return mat;
	}
	void VMaterialManager::MarkDirty(shared_ptr<prime::VMaterialObj> material) {
		dirty_maretial.insert(material->name);
	}

	bool VMaterialManager::ApplyAllPropertiesChanged() {
		if (dirty_maretial.empty()) return false;
		for each (auto name in dirty_maretial)
		{
			auto mat = material_cache[name];
			if (!mat.expired()) {
				mat.lock()->ApplyPropertiesChanged();
			}
		}
		dirty_maretial.clear();
		return true;
	}

	unordered_map<string, weak_ptr<prime::VMeshObj>> VMeshManager::mesh_cache;

	VMesh VMeshManager::Find(const string& name) {
		do
		{
			auto pair = mesh_cache.find(name);
			if (pair != mesh_cache.end()) {
				if (pair->second.expired()) {
					mesh_cache.erase(name);
					break;
				}
				else
				{
					return pair->second.lock();
				}
			}
		} while (false);

		auto mesh = VMesh(new prime::VMeshObj(name));
		mesh_cache[name] = mesh;
		return mesh;
	}



	unordered_map<string, weak_ptr<prime::VDispatchObj>> VDispatchManager::dispatch_cache;

	VDispatch VDispatchManager::Find(const string& name) {
		do
		{
			auto pair = dispatch_cache.find(name);
			if (pair != dispatch_cache.end()) {
				if (pair->second.expired()) {
					dispatch_cache.erase(name);
					break;
				}
				else
				{
					return pair->second.lock();
				}
			}
		} while (false);

		auto dispatch = VDispatch(new prime::VDispatchObj(name));
		dispatch_cache[name] = dispatch;
		return dispatch;
	}
}