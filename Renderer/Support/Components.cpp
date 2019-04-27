#include "Components.h"
#define STB_IMAGE_IMPLEMENTATION
#include "Support/stb_image.h"


static unordered_map<string, VShader*> shader_cache;
static unordered_map<string, VTexture*> texture_cache;
static unordered_map<string, VMaterial*> material_table;
static unordered_map<string, VGeometry*> geometry_cache;

static set<VMaterial*> material_properties_change_table;
static set<VTransform*> transform_change_table;


static Buffer int3_default = NULL;
static Buffer float3_default = NULL;
static Buffer float2_default = NULL;
static Buffer int_default = NULL;


VMaterial * VMaterial::Find(string name) {
	auto pair = material_table.find(name);
	if (pair != material_table.end()) {
		return pair->second;
	}
	auto mat = new VMaterial(name);
	material_table[name] = mat;
	return mat;
}

unordered_map<string, VMaterial*> VMaterial::GetAllMaterials() {
	return material_table;
}

VShader::VShader(string shader_name) {
	LoadFromFile(shader_name);
}

void VShader::LoadFromFile(string shader_name) {
	try {
		for each (auto pair in closestHitProgram)
		{
			pair.second->destroy();
		}
		for each (auto pair in anyHitProgram)
		{
			pair.second->destroy();
		}
		closestHitProgram.clear();
		anyHitProgram.clear();
		auto& context = OptiXLayer::Context();
		string all_contain;
		std::ifstream file((std::string(sutil::samplesDir()) + "/Shaders/" + shader_name + ".cu").c_str());
		if (file.good()) {
			// Found usable source file
			std::stringstream source_buffer;
			source_buffer << file.rdbuf();
			all_contain = source_buffer.str();
		}
		const char *ptx = sutil::getPtxString("Shaders", (shader_name + ".cu").c_str());

		regex r("#pragma (.*?) (.*?)\n");
		sregex_iterator it(all_contain.begin(), all_contain.end(), r);
		sregex_iterator end;
		for (; it != end; ++it)
		{
			int ray_type = Shit<int>::ToProperty(it->operator[](1));
			string program_type = Shit<string>::ToProperty(it->operator[](2));
			if (program_type == "ClosestHit") {
				Program program = context->createProgramFromPTXString(ptx, shader_name + "_ClosestHit");
				closestHitProgram[ray_type] = program;
			}
			else if (program_type == "AnyHit") {
				Program program = context->createProgramFromPTXString(ptx, shader_name + "_AnyHit");
				anyHitProgram[ray_type] = program;
			}
		}
	}
	catch (Exception& e) {
		cout << e.getErrorString() << endl;
		system("PAUSE");
	}
}

void VShader::Release() {
	if (reference.size() != 0) throw Exception("reference is not zero!");
	for each (auto pair in closestHitProgram)
	{
		pair.second->destroy();
	}
	for each (auto pair in anyHitProgram)
	{
		pair.second->destroy();
	}
	closestHitProgram.clear();
	anyHitProgram.clear();
}

VShader * VShader::Find(string name) {
	auto pair = shader_cache.find(name);
	if (pair != shader_cache.end()) {
		return pair->second;
	}
	auto shader = new VShader(name);
	shader_cache[name] = shader;
	return shader;
}

void VShader::Reload(string name) {
	auto pair = shader_cache.find(name);
	if (pair != shader_cache.end()) {
		sutil::ReleasePtxString(name.c_str());
		pair->second->LoadFromFile(name);
		for each (auto mf in pair->second->reference) {
			mf.second();
		}
	}
}

unordered_map<string, VShader*> VShader::GetAllShaders() {
	return shader_cache;
}

VGeometry::VGeometry(string name) {
	auto& context = OptiXLayer::Context();
	// Set up parallelogram programs
	string all_contain;
	std::ifstream file((std::string(sutil::samplesDir()) + "/Geometries/" + name + ".cu").c_str());
	if (file.good()) {
		// Found usable source file
		std::stringstream source_buffer;
		source_buffer << file.rdbuf();
		all_contain = source_buffer.str();
	}
	const char *ptx = sutil::getPtxString("Geometries", (name + ".cu").c_str());

	{
		regex r("#pragma bound (.*?)\n");
		sregex_iterator it(all_contain.begin(), all_contain.end(), r);
		sregex_iterator end;
		int k = 0;
		for (; it != end; ++it) {
			string bound_prog_name = it->operator[](1);
			bound = context->createProgramFromPTXString(ptx, bound_prog_name);
			k++;
		}
		if (k != 1) throw Exception("More than one bound pragma founded in " + name);
	}
	{
		regex r("#pragma intersect (.*?)\n");
		sregex_iterator it(all_contain.begin(), all_contain.end(), r);
		sregex_iterator end;
		int k = 0;
		for (; it != end; ++it) {
			string intersect_prog_name = it->operator[](1);
			intersect = context->createProgramFromPTXString(ptx, intersect_prog_name);
			k++;
		}
		if (k != 1) throw Exception("More than one intersect pragma founded in " + name);
	}
}

VGeometry * VGeometry::Find(string name) {
	auto pair = geometry_cache.find(name);
	if (pair != geometry_cache.end()) {
		return pair->second;
	}
	auto geo = new VGeometry(name);
	geometry_cache[name] = geo;
	return geo;
}



bool VMaterial::ApplyAllChanges()
{
	bool res = false;
	if (material_properties_change_table.size()) res = true;
	for each (auto mat in material_properties_change_table)
	{
		mat->ApplyPropertiesChange();
	}
	material_properties_change_table.clear();
	return res;
}

void VMaterial::ReloadMaterial(string name) {
	auto pair = material_table.find(name);
	if (pair != material_table.end()) {
		pair->second->Reload(name);
		return;
	}
	auto mat = new VMaterial(name);
	material_table[name] = mat;
}

VMaterial::VMaterial(string name) {
	Reload(name);
}

void VMaterial::ReloadShader() {
	if (shader_name == "") return;
	if (mat) mat->destroy();
	mat = OptiXLayer::Context()->createMaterial();
	auto shader = VShader::Find(shader_name);
	for each (auto pair in shader->closestHitProgram)
		mat->setClosestHitProgram(pair.first, pair.second);
	for each (auto pair in shader->anyHitProgram)
		mat->setAnyHitProgram(pair.first, pair.second);
	for each (auto mf in reference) {
		mf.second();
	}
	MarkDirty();
}

void VMaterial::Release() {
	if (reference.size() != 0) throw Exception("reference is not zero!");
	VShader::Find(shader_name)->reference.erase(this);
	for each (auto pair in properties) {
		pair.second.Release();
	}
	properties.clear();
	name = shader_name = "";
}

void VMaterial::Reload(string name) {
	this->name = name;
	mat = OptiXLayer::Context()->createMaterial();
	PropertyReader reader("/Materials", name + ".txt");
	if (shader_name != "") VShader::Find(shader_name)->reference.erase(this);
	shader_name = reader.GetPropertyValue<string>("Shader");
	auto shader = VShader::Find(shader_name);
	properties = reader.GetAllProperties();
	shader->reference[this] = bind(&VMaterial::ReloadShader, this);
	ReloadShader();
	MarkDirty();
}

void VMaterial::SaveToFile() {
	if (name == "" || shader_name == "") return;
	PropertyWriter writer("/Materials", name + ".txt");
	for each (auto pair in properties)
	{
		writer.AddProperty(pair.first, pair.second);
	}
}

void VMaterial::ApplyPropertiesChange() {
	if (!mat) return;
	for each (auto pair in properties)
	{
		if (pair.second.Type() == "string") {
			int k1 = pair.first.find('|');
			if (k1 != -1) {
				string special_type = pair.first.substr(0, k1);
				string name = pair.first.substr(k1 + 1, pair.first.length() - k1 - 1);
				if (special_type == "Texture") {
					int id = VTexture::Find(*pair.second.GetData<string>())->sampler->getId();
					mat[name]->setUserData(sizeof(int), (void*)&id);
				}
			}			
		}
		else {
			pair.second.SetProperty(mat, pair.first);
		}
	}
	SetShaderAsShaderProperties();
}

void VMaterial::SetShaderAsShaderProperties() {
	string new_shader_nam = *properties["Shader"].GetData<string>();
	if (shader_name == new_shader_nam) return;
	if (shader_name != "") VShader::Find(shader_name)->reference.erase(this);
	shader_name = new_shader_nam;
	auto shader = VShader::Find(shader_name);
	shader->reference[this] = bind(&VMaterial::ReloadShader, this);
	ReloadShader();
}

void VMaterial::MarkDirty() {
	material_properties_change_table.insert(this);
}


const static float indentity[16]{ 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };

VTransform::VTransform() {
	auto& context = OptiXLayer::Context();
	transform = context->createTransform();
	transform->setMatrix(false, indentity, indentity);

	pos = rotate = make_float3(0);
	scale = make_float3(1);
}

void VTransform::Setparent(VTransform * trans) {
	if (this == Root()) throw Exception("try to set parent of root transform.");
	if (trans == NULL) trans = Root();
	if (trans == parent) return;
	if (parent) {
		parent->childs.erase(this);
		if (parent == Root()) parent->group->removeChild(transform);
	}
	parent = trans;
	if (parent == Root()) parent->group->addChild(transform);
	parent->childs.insert(this);
}

void VTransform::AddChild(VTransform * trans) {
	trans->Setparent(this);
}

VTransform* VTransform::Root() {
	static VTransform& root = VTransform();
	if (!root.group) {
		root.group = OptiXLayer::Context()->createGroup();
		root.group->setAcceleration(OptiXLayer::Context()->createAcceleration("Trbvh"));
	}
	return &root;
}

void VTransform::MarkDirty() {
	transform_change_table.insert(this);
	if (parent) {
		parent->MarkDirty();
	}
}

bool VTransform::ApplyAllChanges() {
	bool res = false;
	if (transform_change_table.size()) res = true;
	for each (auto trans in transform_change_table)
	{
		trans->ApplyPropertiesChange();
	}
	transform_change_table.clear();
	return res;
}


VGeometryFilter::VGeometryFilter(VGeometry * geometry) {
	auto& context = OptiXLayer::Context();
	this->geometry = context->createGeometry();
	geometry_shader = geometry;
	this->geometry->setBoundingBoxProgram(geometry_shader->bound);
	this->geometry->setIntersectionProgram(geometry_shader->intersect);
	this->geometry->setPrimitiveCount(1u);
}

Handle<VariableObj> VGeometryFilter::Visit(const char * varname)
{
	object->MarkDirty();
	return geometry[varname];
}

void VGeometryFilter::SetMesh(VMesh * mesh) {
	object->MarkDirty();

	geometry["vertex_buffer"]->setBuffer(mesh->vert_buffer);
	geometry["v_index_buffer"]->setBuffer(mesh->v_index_buffer);


	if (float3_default == NULL) {
		float3_default = OptiXLayer::Context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0);
		int3_default = OptiXLayer::Context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, 0);
		float2_default = OptiXLayer::Context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);
		int_default = OptiXLayer::Context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 0);
	}

	if (mesh->normal_buffer != NULL) {
		geometry["normal_buffer"]->setBuffer(mesh->normal_buffer);
		geometry["n_index_buffer"]->setBuffer(mesh->n_index_buffer);
	}
	else {
		geometry["normal_buffer"]->setBuffer(float3_default);
		geometry["n_index_buffer"]->setBuffer(int3_default);
	}
	if (mesh->tex_buffer != NULL) {
		geometry["texcoord_buffer"]->setBuffer(mesh->tex_buffer);
		geometry["t_index_buffer"]->setBuffer(mesh->t_index_buffer);
	}
	else {
		geometry["texcoord_buffer"]->setBuffer(float2_default);
		geometry["t_index_buffer"]->setBuffer(int3_default);
	}

	geometry["material_buffer"]->setBuffer(int_default); //todo

	RTsize size = -1; mesh->v_index_buffer->getSize(size);
	geometry->setPrimitiveCount(size);
}

void VObject::RebindMaterial() {
	go->setMaterial(0, material->mat);
}

VObject::VObject(string geometry_shader_name) {
	auto& context = OptiXLayer::Context();
	go = context->createGeometryInstance();
	transform = new VTransform();
	transform->Setparent();


	hook = context->createGeometryGroup();
	hook->setAcceleration(context->createAcceleration("Trbvh"));
	hook->addChild(go);

	transform->transform->setChild(hook);
	transform->object = this;

	auto geo = VGeometry::Find(geometry_shader_name);
	geometryFilter = new VGeometryFilter(geo);
	geometryFilter->object = this;
	go->setGeometry(geometryFilter->geometry);
}

void VObject::SetMaterial(VMaterial * mat) {
	if (mat == material) return;
	if (material) { material->reference.erase(this); }
	if (!mat) material = VMaterial::Find("default");
	else material = mat;
	go->setMaterialCount(1);
	go->setMaterial(0, material->mat);
	material->reference[this] = bind(&VObject::RebindMaterial, this);
}


unordered_map<string, VMesh*> mesh_table;

VMesh::VMesh(string name) {
	int k = name.find_last_of('.');
	if (k == -1) throw Exception("ERROR mesh name.");

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
		tinyobj::LoadObj(&attrib, &shapes, NULL, &warn, &err, (string(sutil::samplesDir()) + "/Meshs/" + name).c_str());
		if (err.length() != 0) throw Exception("ERROR read mesh.\n" + err);
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
			for (int i = 0; i <shapes[j].mesh.indices.size() / 3; i++)
			{
				int3 v;
				v.x = shapes[j].mesh.indices[i * 3].vertex_index;
				v.y = shapes[j].mesh.indices[i * 3 + 1].vertex_index;
				v.z = shapes[j].mesh.indices[i * 3 + 2].vertex_index;
				v_index.push_back(v);
			}
			for (int i = 0; i <shapes[j].mesh.indices.size() / 3; i++)
			{
				int3 v;
				v.x = shapes[j].mesh.indices[i * 3].texcoord_index;
				v.y = shapes[j].mesh.indices[i * 3 + 1].texcoord_index;
				v.z = shapes[j].mesh.indices[i * 3 + 2].texcoord_index;
				t_index.push_back(v);
			}
			for (int i = 0; i <shapes[j].mesh.indices.size() / 3; i++)
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


	auto& context = OptiXLayer::Context();

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

VMesh::~VMesh() {
	SAFE_RELEASE_OPTIX_OBJ(vert_buffer);
	SAFE_RELEASE_OPTIX_OBJ(normal_buffer);
	SAFE_RELEASE_OPTIX_OBJ(tex_buffer);
	SAFE_RELEASE_OPTIX_OBJ(v_index_buffer);
	SAFE_RELEASE_OPTIX_OBJ(n_index_buffer);
	SAFE_RELEASE_OPTIX_OBJ(t_index_buffer);
}

VMesh * VMesh::Find(string name) {
	auto pair = mesh_table.find(name);
	if (pair != mesh_table.end()) {
		return pair->second;
	}
	auto mat = new VMesh(name);
	mesh_table[name] = mat;
	return mat;
}

Buffer default_tex = NULL;
TextureSampler default_tex_sampler = NULL;

VTexture::VTexture(string path) {
	auto& context = OptiXLayer::Context();


	if (path == "") {
		if (default_tex == NULL) {
			default_tex = context->createBuffer(RT_BUFFER_INPUT);
			default_tex->setFormat(RT_FORMAT_UNSIGNED_BYTE4);
			default_tex->setSize(1, 1);

			default_tex_sampler = context->createTextureSampler();
			default_tex_sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
			default_tex_sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
			default_tex_sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
			default_tex_sampler->setReadMode(RT_TEXTURE_READ_ELEMENT_TYPE);
			default_tex_sampler->setMaxAnisotropy(1);
			default_tex_sampler->setMipLevelClamp(0, 1);
			default_tex_sampler->setArraySize(1);
			default_tex_sampler->setBuffer(default_tex);
		}
		sampler = default_tex_sampler;
	}
	else {
		buffer = context->createBuffer(RT_BUFFER_INPUT);
		int w, h, n;
		auto img = stbi_load((string(sutil::samplesDir()) + "/Textures/" + path).c_str(), &w, &h, &n, STBI_rgb_alpha);

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
	}

}

VTexture::~VTexture() {
	SAFE_RELEASE_OPTIX_OBJ(buffer);
	if (sampler != default_tex_sampler && sampler != NULL) sampler->destroy();
}

VTexture * VTexture::Find(string path) {
	auto pair = texture_cache.find(path);
	if (pair != texture_cache.end()) {
		return pair->second;
	}
	auto tex = new VTexture(path);
	texture_cache[path] = tex;
	return tex;
}
