#pragma once


#include "../Support/Basic.h"

#include <unordered_map>
#include <sstream>
#include <regex>
#include <unordered_set>

namespace VRender {

	namespace prime {

		struct ProgramWrapper {
			optix::Program program;
			ProgramWrapper(Program& program) {this->program = program; }
			~ProgramWrapper() { SAFE_RELEASE_OPTIX_OBJ(program); }
		};
		typedef shared_ptr<ProgramWrapper> VProgram;



		class VShaderObj {
			unordered_map<int, VProgram> closestHitPrograms, anyHitPrograms;
			string name;
			void LoadFromFile(const string& shader_name);
		public:
			VShaderObj(const string& name) { LoadFromFile(name); this->name = name; }

			void ApplyShader(optix::Material mat);
		};
		typedef shared_ptr<VShaderObj> VShader;



		class VTextureObj {
			optix::Buffer buffer = NULL;
			optix::TextureSampler sampler = NULL;
			int id = 0;
		public:
			const int& ID() { return id; }
			VTextureObj(const string& path);
			~VTextureObj();
		};
		typedef shared_ptr<VTextureObj> VTexture;



		class VMaterialObj {
			VShader shader;
			unordered_map<string, VTexture> textures;
			unordered_map<string, VProperty> properties;

		public:
			string name;
			optix::Material material;

			void Load(const string& name);
		    
			void Save() { }

			VMaterialObj(const string& name) {
				material = OptixInstance::Instance().Context()->createMaterial();
				Load(name);
			}

			unordered_map<string, VProperty>& Properties() {
				return properties;
			}

			void ApplyPropertiesChanged() {

				material->setAnyHitProgram(0, nullptr);
				material->setAnyHitProgram(1, nullptr);
				material->setClosestHitProgram(0, nullptr);
				material->setClosestHitProgram(1, nullptr);

				shader->ApplyShader(material);

				while (material->getVariableCount() != 0)
				{
					material->removeVariable(material->getVariable(0));
				}

				for each (auto pair in properties)
				{
					if (pair.second.Type() == "string") {
						int k1 = pair.first.find('|');
						if (k1 != -1) {
							string special_type = pair.first.substr(0, k1);
							string name = pair.first.substr(k1 + 1, pair.first.length() - k1 - 1);
							if (special_type == "Texture") {
								int id = textures[*pair.second.GetData<string>()]->ID();
								material[name]->setUserData(sizeof(int), (void*)& id);
							}
						}
					}
					else {
						pair.second.SetProperty(material, pair.first);
					}
				}
			}

			~VMaterialObj() {
				material->destroy();
				textures.clear();
				for each (auto pair in properties)
					pair.second.Release();
			}
		};
		typedef shared_ptr<VMaterialObj> VMaterial;



		struct VMeshObj{
			string name;
			Buffer vert_buffer = NULL;
			Buffer normal_buffer = NULL;
			Buffer tex_buffer = NULL;
			Buffer v_index_buffer = NULL;
			Buffer n_index_buffer = NULL;
			Buffer t_index_buffer = NULL;

			VMeshObj(const string& name);

			~VMeshObj();
		};
		typedef shared_ptr<VMeshObj> VMesh;



		class VDispatchObj {
		public:
			optix::Program rayGenerationProgram, exceptionProgram, missProgram;
			string name;
		protected:
			void LoadFromFile(const string& shader_name) {
				try {
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

					rayGenerationProgram = context->createProgramFromPTXString(ptx, "dispatch");
					exceptionProgram = context->createProgramFromPTXString(ptx, "exception");
					missProgram = context->createProgramFromPTXString(ptx, "miss");
				}
				catch (Exception & e) {
					cout << e.getErrorString() << endl;
					system("PAUSE");
				}
			};
		public:
			VDispatchObj(const string& name) { LoadFromFile(name); this->name = name; }
		};
		typedef shared_ptr<VDispatchObj> VDispatch;
	}

	typedef prime::VShader VShader;

	class VShaderManager {
		static unordered_map<string, weak_ptr<prime::VShaderObj>> shader_cache;
	public:
		static VShader Find(const string& name);
	};



	typedef prime::VTexture VTexture;

	class VTextureManager {
		static unordered_map<string, weak_ptr<prime::VTextureObj>> texture_cache;
	public:
		static VTexture Find(const string& name);
	};
	


	typedef prime::VMaterial VMaterial;

	class VMaterialManager {
		static unordered_map<string, weak_ptr<prime::VMaterialObj>> material_cache;
		static unordered_set<string> dirty_maretial;
	public:
		static VMaterial Find(const string& name);
		static void MarkDirty(shared_ptr<prime::VMaterialObj> material);
		static bool ApplyAllPropertiesChanged();
	};

	typedef prime::VMesh VMesh;

	class VMeshManager {
		static unordered_map<string, weak_ptr<prime::VMeshObj>> mesh_cache;
	public:
		static VMesh Find(const string& name);
	};

	typedef  prime::VDispatch VDispatch;

	class VDispatchManager {
		static unordered_map<string, weak_ptr<prime::VDispatchObj>> dispatch_cache;
	public:
		static VDispatch Find(const string& name);
	};

	class VResources {
	public:
		template<typename T>
		static T Find(const string& name) { throw Exception("Unknow resources type!"); };
		template<>
		static VShader Find<VShader>(const string& name) { return VShaderManager::Find(name); };
		template<>
		static VTexture Find<VTexture>(const string& name) { return VTextureManager::Find(name); };
		template<>
		static VMaterial Find<VMaterial>(const string& name) { return VMaterialManager::Find(name); };
		template<>
		static VMesh Find<VMesh>(const string& name) { return VMeshManager::Find(name); };
		template<>
		static VDispatch Find<VDispatch>(const string& name) { return VDispatchManager::Find(name); };
	};
}