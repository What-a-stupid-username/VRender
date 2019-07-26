#pragma once


#include "RenderView.hpp"

#include <shellapi.h>
#include <dwmapi.h>
#include "Imgui/imgui.h"
#include "Imgui/imgui_stdlib.h"
#include "Imgui/imgui_impl_win32.h"
#include "Imgui/imgui_impl_dx12.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <tchar.h>
#include <time.h>
#include<io.h>

template<bool file>
class FileLoader {
private:
	void find(const char* lpPath, string type)
	{
		vector<string> files_new;
		files.clear();
		floders.clear();

		char szFind[MAX_PATH];
		WIN32_FIND_DATA FindFileData;

		strcpy(szFind, lpPath);
		strcat(szFind, "/*.*");

		HANDLE hFind = ::FindFirstFile(szFind, &FindFileData);
		if (INVALID_HANDLE_VALUE == hFind)  return;

		while (true)
		{
			if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				if (FindFileData.cFileName[0] != '.')
				{
					floders.push_back((char*)(FindFileData.cFileName));
				}
			}
			else
			{
				files_new.push_back(FindFileData.cFileName);
			}
			if (!FindNextFile(hFind, &FindFileData))  break;
		}
		FindClose(hFind);

		for each (auto name in files_new)
		{
			int n_l = name.length(), t_l = type.length();
			if (n_l > t_l) {
				if (name.substr(n_l - t_l, t_l) == type) {
					files.push_back(name);
				}
			}
		}

	}
	vector<string> files;
	vector<string> floders;
public:
	string result;

public:
	FileLoader() { ResetToDefault(); }
	void ResetToDefault() {
		result = sutil::samplesDir();
	}

	bool Draw(ImVec2& need_size, string type, bool& open) {
		if (!open) return false;
		ImGui::Begin("Setlect file", &open, ImGuiWindowFlags_::ImGuiWindowFlags_NoResize | ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse);
		ImGui::SetWindowFocus();
		int l = min<int>(max<int>(7 * result.size() + 30, 200), 800);
		ImGui::SetWindowSize(ImVec2(l, 250));

		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		need_size = ImVec2(max(need_size.x, tmp.x), max(need_size.y, tmp.y));

		find(result.c_str(), type);

		ImGui::Text(result.c_str());
		ImGui::Separator();

		int k = result.find_last_of('/');
		if (k != -1) {
			if (ImGui::Selectable("..###FileLoader", false, ImGuiSelectableFlags_::ImGuiSelectableFlags_AllowDoubleClick)) {
				result = result.substr(0, k);
			}
		}


		int index = 0;

		for each (auto name in files)
		{
			if (ImGui::Selectable((name + "###FileLoader" + to_string(index++)).c_str(), false, ImGuiSelectableFlags_::ImGuiSelectableFlags_AllowDoubleClick)) {
				if (file) result += "/" + name;
				open = false;
				ImGui::End();
				return true;
			}
		}

		for each (auto name in floders)
		{
			if (ImGui::Selectable(("/" + name + "###FileLoader" + to_string(index++)).c_str(), false, ImGuiSelectableFlags_::ImGuiSelectableFlags_AllowDoubleClick)) {
				result += "/" + name;
			}
		}
		ImGui::End();
		return false;
	}
};


class VGUI {
	VRender::VRenderer& renderer;

	bool show_console_window = true;
	bool show_setting_window = false;
	bool show_output_window = false;

	ImVec2 next_Window_pos;
	ImVec2 next_Window_pos_2;

	int auto_resize = 2;

	int selected_obj = -1;

public:
	ImVec4 back_ground_color = ImVec4(1.f, 0.55f, 1.f, 1.f);
	ImVec2 needed_size;


private:
	//void OpenMaterialFile(string name) {
	//	string str = ((string(sutil::samplesDir()) + "/Materials/" + name + ".txt").c_str());
	//	auto k = ShellExecute(DX_Window, "open", str.c_str(), NULL, NULL, SW_SHOW);
	//	//thread([name]() {cout << ; }).detach();d
	//}
	//void OpenShaderFile(string name) {
	//	string str = ((string(sutil::samplesDir()) + "/Shaders/" + name + ".cu").c_str());
	//	auto k = ShellExecute(DX_Window, "open", str.c_str(), NULL, NULL, SW_SHOW);
	//	//thread([name]() {cout << ; }).detach();d
	//}

	void LeftLabelText(string label, string txt) {
		ImGui::Text(label.c_str()); ImGui::SameLine();
		ImGui::Text(txt.c_str());
	}

	void DrawMainMenu() {
		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin(VRender::VScene::SceneName().c_str(), 0, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoResize);
		ImGui::SetWindowPos(ImVec2(0, 0));
		ImGui::SetWindowSize(ImVec2(200, 180));
		next_Window_pos = ImGui::GetWindowPos() + ImVec2(0, ImGui::GetWindowHeight());
		next_Window_pos_2 = ImGui::GetWindowPos() + ImVec2(ImGui::GetWindowWidth(), 0);
		needed_size = ImGui::GetWindowSize();

		static float fps = 0;
		int frame = renderer.GlobalFrameNumber();
		{
			static int last_time = 0;
			static int last_frame = 0;
			int delta_time;
			if ((delta_time = clock() - last_time) > 100) {
				fps = float(frame - last_frame) / delta_time * 1000;
				last_time += delta_time;
				last_frame = frame;
			}
		}

		ImGui::Text("Render Time:");
		ImGui::Text(" %.1f ms(%.0f FPS)", 1000.f / fps, fps);
		ImGui::Text("Frame No.%d", renderer.Camera()->staticFrameNum);
		ImGui::Separator();

		ImGui::Checkbox("Open Console", &show_console_window);
		ImGui::Checkbox("Open Setting Window", &show_setting_window);
		ImGui::Checkbox("Open Output Window", &show_output_window);

		ImGui::End();
	}

	void DrawConsole() {
		DrawWindowRightColum("Console", 100, &show_console_window);

		static bool pause = false;
		if (pause) {
			if (ImGui::Button("Continue Render")) {
				renderer.EnableRenderer(true);
				pause = !pause;
			}
		}
		else {
			if (ImGui::Button("Pause Render")) {
				renderer.EnableRenderer(false);
				pause = !pause;
			}
		}

		static FileLoader<false> loader;
		static bool loader_open = false;
		static bool loading = false;
		if (ImGui::Button("Load scene")) { loader_open = true; }
		if (loader.Draw(needed_size, "txt", loader_open)) {
			sutil::ClearPtxCache();
			VRender::VScene::percentage = 0;
			ImGui::OpenPopup("LoadingScene");
			thread([&]() {
				string path = loader.result;
				loading = true;
				renderer.Lock();
				if (!VRender::VScene::LoadScene(path)) {
					VRender::VScene::LoadScene(string(sutil::samplesDir()) + "/Cornell");
				}
				loading = false;
				renderer.Unlock();
			}).detach();
		}
		if (loading) {
			if (ImGui::BeginPopupModal("LoadingScene", NULL, ImGuiWindowFlags_::ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_::ImGuiWindowFlags_NoResize | ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse))
			{
				float v = VRender::VScene::percentage;
				ImGui::ProgressBar(v, ImVec2(200, 20));
				ImGui::EndPopup();
			}
		}
		float fov = renderer.Camera()->fov.x;
		if (ImGui::DragFloat("Camera Fov", &fov, 1, 0, 90, "%.1f")) {
			renderer.Camera()->fov = make_float2(fov);
			renderer.Camera()->staticFrameNum = 0;
		}

		ImGui::End();
	}

	void DrawInspector() {
		if (selected_obj <= -1) return;
		DrawWindowRightColum("Inspector", 400);

		VRender::VObject obj = VRender::VObjectManager::GetObjectByID(selected_obj);

		int light_id = obj->light_id;
		if (light_id != -1) { // Light
			VRender::VLight light = VRender::VLightManager::GetLight(light_id);
			if (ImGui::InputText("Name", &(light->name))) {
				obj->name = light->name;
			}
			ImGui::Separator();

			string light_id_str = to_string(light_id);
			bool changed = false;
			ImGui::Text("Transform");
			changed |= ImGui::DragFloat3(("Position###pos_input" + light_id_str).c_str(), (float*)&(light->position), 0.1);
			changed |= ImGui::DragFloat3(("Rotation###rot_input" + light_id_str).c_str(), (float*) & (light->rotation), 0.1);
			changed |= ImGui::DragFloat3(("Scale###sca_input" + light_id_str).c_str(), (float*) & (light->scale), 0.01, 0);
			ImGui::Separator();
			ImGui::Text("Light");
			changed |= ImGui::ColorEdit3(("Color###color_input" + light_id_str).c_str(), (float*)&(light->color));
			changed |= ImGui::DragFloat(("Emission###color_input" + light_id_str).c_str(), &light->emission, 1, 0, 100);
			if (changed) {
				VRender::VLightManager::MarkDirty(light_id);
			}
		}
		else {
			string id_str = to_string(selected_obj);
			{
				ImGui::InputText("Name", &(obj->name));
				ImGui::Separator();
			}
			{

				ImGui::Text("Transform");
				auto trans = obj->Transform();
				bool changed = false;
				changed |= ImGui::DragFloat3(("Position###pos_input" + id_str).c_str(), trans->Position<float>(),0.1);
				changed |= ImGui::DragFloat3(("Rotation###rot_input" + id_str).c_str(), trans->Rotation<float>(), 0.1);
				changed |= ImGui::DragFloat3(("Scale###sca_input" + id_str).c_str(), trans->Scale<float>(), 0.01, 0);
				if (changed) {
					trans->MarkDirty();
				}
				ImGui::Separator();
			}
			{
				ImGui::Text("MeshRenderer");
				auto mat = obj->MeshRenderer()->GetMaterial();

				bool changed = false;

				int index = 0;
				auto& properties = mat->Properties();
				for each (auto pair in properties)
				{
					if (pair.second.Type() == "int") {
						changed |= ImGui::InputInt((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<int>());
					}
					else if (pair.second.Type() == "string") {
						if (pair.first == "Shader") {
							LeftLabelText(pair.first + ":  ", pair.second.GetData<string>()->c_str());
						}
						else {
							ImGui::InputText((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<string>());
						}
					}
					else if (pair.second.Type() == "float") {
						changed |= ImGui::DragFloat((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<float>(), 0.001, 0, 1);
					}
					else if (pair.second.Type() == "float2") {
						changed |= ImGui::DragFloat2((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<float>(), 0.001, 0, 1);
					}
					else if (pair.second.Type() == "float3") {
						if (pair.first[0] == '!') {
							changed |= ImGui::InputFloat3((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<float>());
						}
						else {
							changed |= ImGui::ColorEdit3((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<float>());
						}
					}
					else if (pair.second.Type() == "float4") {
						if (pair.first[0] == '!') {
							changed |= ImGui::InputFloat4((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<float>());

						}
						else {
							changed |= ImGui::ColorEdit4((pair.first + "###Mat_prop" + to_string(index)).c_str(), pair.second.GetData<float>());
						}
					}
					index++;
				}
				if (changed) {
					VRender::VMaterialManager::MarkDirty(mat);
				}
				ImGui::Separator();
			}
			{
				ImGui::Text("MeshFilter");
				LeftLabelText("Mesh:  ", obj->MeshFilter()->GetMesh()->name);
			}
		}


		ImGui::End();
	}


	bool right_click_on_item;
	void DrawHierarchy() {
		DrawWindowLeftColum("Hierarchy", 400);
			   
		auto objects = VRender::VObjectManager::GetAllObjects();
		
		static bool clicked_here = false;
		if (ImGui::IsMouseHoveringWindow() && ImGui::IsMouseClicked(0)) {
			clicked_here = true;
		}
		else if (!ImGui::IsMouseHoveringWindow() && ImGui::IsMouseClicked(0)) {
			clicked_here = false;
		}

		if (clicked_here && ImGui::IsMouseHoveringWindow() && ImGui::IsMouseReleased(0)) {
			selected_obj = -1;
		}

		right_click_on_item = false;
		for each (auto obj in objects)
		{
			DisplayNode(obj);
		}

		if (ImGui::IsMouseHoveringWindow() && ImGui::IsMouseReleased(1) && !right_click_on_item) {
			ImGui::OpenPopup("CreateObject");
		}
		if (ImGui::BeginPopup("EditObject"))
		{
			string names[3] = { "Delete","Copy","Paste" };
			for (int i = 0; i < 3; i++)
				if (ImGui::Selectable(names[i].c_str())) {
					switch (i)
					{
					case 0:
						break;
					case 1:
						break;
					case 2:
						break;
					default:
						break;
					}
				}
			ImGui::EndPopup();
		}

		if (ImGui::BeginPopup("CreateObject"))
		{
			string names[3] = { "Create", "XX", "XX"};
			for (int i = 0; i < 3; i++)
				if (ImGui::Selectable(names[i].c_str())) {
					switch (i)
					{
					case 0:
						break;
					case 1:
						break;
					case 2:
						break;
					default:
						break;
					}
				}
			ImGui::EndPopup();
		}

		ImGui::End();
	}

	void DisplayNode(std::pair<const int, VRender::VObject> pair) {
		string id_str = to_string(pair.first);
		bool selected = false;
		if (pair.first == selected_obj) selected = true;
		if (ImGui::Selectable((pair.second->name + "###Object" + id_str).c_str(), &selected)) {
			selected_obj = pair.first;
		}
		if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(1)) {
			selected_obj = pair.first;
			right_click_on_item = true;
			ImGui::OpenPopup("EditObject");
		}
	}

	void DrawWindowRightColum(string name, int height, bool* show = NULL) {
		ImGui::Begin(name.c_str(), show, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
		ImGui::SetWindowPos(next_Window_pos_2);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(250, height));
		
		next_Window_pos_2 = next_Window_pos_2 + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
	}

	void DrawWindowLeftColum(string name, int height, bool* show = NULL) {
		ImGui::Begin(name.c_str(), show, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(200, height));

		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
	}

	//void DrawSettings() {
	//	ImGui::Begin("Settings", &show_setting_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
	//	ImGui::SetWindowPos(next_Window_pos);
	//	if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 160));

	//	next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
	//	auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
	//	needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
	//	float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
	//	next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;

	//	bool auto_resize_b = auto_resize != 2;
	//	ImGui::Checkbox("Free size window", &auto_resize_b);
	//	if (auto_resize_b) auto_resize = 0;
	//	else auto_resize = 2;

	//	static int style_idx = 0;
	//	if (ImGui::Combo("Style###StyleSelector", &style_idx, "Dark\0Light\0"))
	//	{
	//		switch (style_idx)
	//		{
	//		case 0: ImGui::StyleColorsDark(); break;
	//		case 1: ImGui::StyleColorsLight(); break;
	//		}
	//	}

	//	ImGuiIO& io = ImGui::GetIO();
	//	ImGui::SliderFloat("Scale", &io.FontGlobalScale, 0.5, 2, "%.1f");


	//	ImGui::End();
	//}

	//void DrawOutput() {
	//	ImGui::Begin("Output", &show_output_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

	//	ImGui::SetWindowPos(next_Window_pos);
	//	if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 100));

	//	next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
	//	float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
	//	next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;

	//	auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
	//	needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

	//	static string name = "VRenderer Output";
	//	ImGui::InputText("Output name", &name[0], name.size() + 1, ImGuiInputTextFlags_::ImGuiInputTextFlags_NoHorizontalScroll | ImGuiInputTextFlags_::ImGuiInputTextFlags_AutoSelectAll);

	//	if (ImGui::Button("Save")) {
	//		thread([&]() { layer.SaveResultToFile(name); }).detach();
	//	}

	//	ImGui::End();
	//}

	//void DrawShader() {
	//	ImGui::Begin("Shaders", &show_shader_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

	//	ImGui::SetWindowPos(next_Window_pos);
	//	auto & shader_table = VShader::GetAllShaders();
	//	if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 80 * (shader_table.size() > 5 ? 5 : (shader_table.size() < 1 ? 1 : shader_table.size()))));

	//	next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
	//	float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
	//	next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;

	//	auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
	//	needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

	//	int index = 0;
	//	for each (auto shader in shader_table)
	//	{
	//		ImGui::Text(("Shader: " + shader.first).c_str());
	//		if (ImGui::Button(("Open Shader###shader_open_" + to_string(index)).c_str())) {
	//			OpenShaderFile(shader.first);
	//		}
	//		ImGui::SameLine();
	//		if (ImGui::Button(("Reload Shader###shader_reload_" + to_string(index)).c_str())) {
	//			layer.Lock();
	//			VShader::Reload(shader.first);
	//			layer.MarkDirty();
	//			layer.Unlock();
	//		}
	//		ImGui::Separator();
	//		index++;
	//	}

	//	ImGui::End();
	//}

	//void DrawMaterial() {
	//	ImGui::Begin("Material", &show_material_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

	//	ImGui::SetWindowPos(next_Window_pos_2);

	//	auto & mat_table = VMaterial::GetAllMaterials();

	//	if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(400, 200 * (mat_table.size() > 3 ? 3 : (mat_table.size() < 1 ? 1 : mat_table.size()))));

	//	next_Window_pos_2 = next_Window_pos_2 + ImVec2(0, ImGui::GetWindowHeight());
	//	auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
	//	needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

	//	int index = 0;
	//	for each (auto mat in mat_table)
	//	{
	//		ImGui::Text(mat.first.c_str());
	//		if (ImGui::Button(("Open File###" + to_string(index)).c_str())) {
	//			OpenMaterialFile(mat.first);
	//		}
	//		ImGui::SameLine();
	//		if (ImGui::Button(("Reload file###1000" + to_string(index)).c_str())) {
	//			layer.Lock();
	//			VMaterial::ReloadMaterial(mat.first);
	//			layer.Unlock();
	//		}
	//		ImGui::SameLine();
	//		if (ImGui::Button(("Save to file###2000" + to_string(index)).c_str())) {
	//			mat.second->SaveToFile();
	//		}

	//		auto propertie = mat.second->GetAllProperties();
	//		int index2 = 0;
	//		for each (auto prop in propertie)
	//		{
	//			if (prop.second.Type() == "int") {
	//				if (ImGui::InputInt((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<int>())) {
	//					mat.second->MarkDirty();
	//				}
	//			}
	//			else if (prop.second.Type() == "string") {
	//				if (prop.first == "Shader") {
	//					LeftLabelText(prop.first + ":  ", prop.second.GetData<string>()->c_str());
	//				}
	//				else {
	//					ImGui::InputText((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<string>());
	//				}
	//			}
	//			else if (prop.second.Type() == "float") {
	//				if (ImGui::DragFloat((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>(), 0.001, 0, 1)) {
	//					mat.second->MarkDirty();
	//				}
	//			}
	//			else if (prop.second.Type() == "float2") {
	//				if (ImGui::DragFloat2((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>(), 0.001, 0, 1)) {
	//					mat.second->MarkDirty();
	//				}
	//			}
	//			else if (prop.second.Type() == "float3") {
	//				if (prop.first[0] == '!') {
	//					if (ImGui::InputFloat3((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
	//						mat.second->MarkDirty();
	//					}
	//				}
	//				else {
	//					if (ImGui::ColorEdit3((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
	//						mat.second->MarkDirty();
	//					}
	//				}
	//			}
	//			else if (prop.second.Type() == "float4") {
	//				if (prop.first[0] == '!') {
	//					if (ImGui::InputFloat4((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
	//						mat.second->MarkDirty();
	//					}
	//				}
	//				else {
	//					if (ImGui::ColorEdit4((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
	//						mat.second->MarkDirty();
	//					}
	//				}
	//			}
	//			index2++;
	//		}

	//		ImGui::Separator();
	//		index++;
	//	}

	//	ImGui::End();
	//}


public:
	VGUI() : renderer(VRender::VRenderer::Instance()) {
		auto& style = ImGui::GetStyle();
		style.FrameRounding = 12.f;
		style.GrabRounding = 12.f;
	}


	void OnDrawGUI() {

		// Left column
		DrawMainMenu();
		
		int tp_selected_obj = selected_obj = renderer.GetSelectedObject();		
		DrawHierarchy();
		if (selected_obj != tp_selected_obj) renderer.SetSelectedObject(selected_obj);

		// Right column
		if (show_console_window) DrawConsole();
		DrawInspector();
		
	}
};