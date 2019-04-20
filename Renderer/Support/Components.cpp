#include "Components.h"


static unordered_map<string, VShader*> shader_cache;
static unordered_map<string, VMaterial*> material_table;
static unordered_map<string, VGeometry*> geometry_cache;

static set<VMaterial*> material_properties_change_table;
static set<VTransform*> transform_change_table;



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



bool VMaterial::ApllyAllChanges()
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
		pair.second.SetProperty(mat, pair.first);
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