#ifndef GUI_HPP_
#define GUI_HPP_

#include <GlutEvent.h>
#include <Support/Components.h>
#include <Support/Scene.h>

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
				result += "/" + name;
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
	OptiXLayer& layer;

	bool show_setting_window = false;
	bool show_console_window = true;
	bool show_output_window = false;
	bool show_material_window = false;
	bool show_shader_window = false;
	bool show_scene_window = false;

	ImVec2 next_Window_pos;
	ImVec2 next_Window_pos_2;

	int auto_resize = 0;

public:
	ImVec4 back_ground_color = ImVec4(1.f, 0.55f, 1.f, 1.f);
	ImVec2 needed_size;


private:
	void OpenMaterialFile(string name) {
		string str = ((string(sutil::samplesDir()) + "/Materials/" + name + ".txt").c_str());
		auto k = ShellExecute(DX_Window, "open", str.c_str(), NULL, NULL, SW_SHOW);
		//thread([name]() {cout << ; }).detach();d
	}
	void OpenShaderFile(string name) {
		string str = ((string(sutil::samplesDir()) + "/Shaders/" + name + ".cu").c_str());
		auto k = ShellExecute(DX_Window, "open", str.c_str(), NULL, NULL, SW_SHOW);
		//thread([name]() {cout << ; }).detach();d
	}

	void LeftLabelText(string label, string txt) {
		ImGui::Text(label.c_str()); ImGui::SameLine();
		ImGui::Text(txt.c_str());
	}

	void DrawMainMenu() {
		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin("Main Menu", 0, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | 
										ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
										ImGuiWindowFlags_::ImGuiWindowFlags_NoResize);
		ImGui::SetWindowPos(ImVec2(0, 0));
		ImGui::SetWindowSize(ImVec2(200, 240));
		next_Window_pos = ImGui::GetWindowPos() + ImVec2(ImGui::GetWindowWidth(), 0);
		next_Window_pos_2 = next_Window_pos;
		needed_size = ImGui::GetWindowSize();

		static float fps = 0;
		int frame = OptiXLayer::Camera().StaticFrameNum();
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
		ImGui::Text("Render Time: %.1f ms(%.0f FPS)", 1000.f / fps, fps);
		ImGui::Text("Frame No.%d", frame);
		ImGui::Separator();

		ImGui::Checkbox("Open Console", &show_console_window);
		ImGui::Checkbox("Open Setting Window", &show_setting_window);
		ImGui::Checkbox("Open Output Window", &show_output_window);
		ImGui::Checkbox("Open Material Window", &show_material_window);
		ImGui::Checkbox("Open Shader View", &show_shader_window);
		ImGui::Checkbox("Open Scene Window", &show_scene_window);

		ImGui::End();
	}

	void DrawConsole() {
		ImGui::Begin("Console", &show_console_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300,240));

		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
		float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
		next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;
		
		if (layer.pause) {
			if (ImGui::Button("Continue Renderer")) layer.pause = false;
		}
		else {
			if (ImGui::Button("Pause Renderer")) layer.pause = true;
		}

		if (ImGui::Button("Reload scene")) {
			sutil::ClearPtxCache();
			layer.LoadScene();
		}

		ImGui::Separator();
		
		if (ImGui::CollapsingHeader("Result Type"))
		{
			if (ImGui::MenuItem("original Buffer", 0, layer.resultType == OptiXLayer::ResultBufferType::origial)) 
				layer.resultType = OptiXLayer::ResultBufferType::origial;
			if (ImGui::MenuItem("tonmap Buffer", 0, layer.resultType == OptiXLayer::ResultBufferType::tonemap)) 
				layer.resultType = OptiXLayer::ResultBufferType::tonemap;
			if (ImGui::MenuItem("denoised Buffer", 0, layer.resultType == OptiXLayer::ResultBufferType::denoise))
				layer.resultType = OptiXLayer::ResultBufferType::denoise;
			if (ImGui::MenuItem("helper Buffer", 0, layer.resultType == OptiXLayer::ResultBufferType::helper))
				layer.resultType = OptiXLayer::ResultBufferType::helper;
		}

		ImGui::Separator();

		{
			bool k = ImGui::SliderFloatLableOnLeft("Diffuse strength", "", &layer.diffuse_strength, 0, 10, "%.2f");
			if (k) layer.MarkDirty();
		}

		static bool post = false;
		if (layer.resultType != OptiXLayer::ResultBufferType::origial && layer.resultType != OptiXLayer::ResultBufferType::helper) {
			if (post == false) layer.RebuildCommandList(true);
			ImGui::SliderFloatLableOnLeft("Exposure", "###1", &layer.exposure, 0, 100, "%.4f", 3);
			post = true;
		}
		else {
			if (post == true) layer.RebuildCommandList(false);
			post = false;
		}

		ImGui::Separator();

		{
			bool k = ImGui::SliderInt("###2", &layer.max_depth, 0, 10, "max tracing depth = %d");
			if (k) layer.MarkDirty();
		}

		{
			int sample_num_per_pixel = layer.sqrt_num_samples * layer.sqrt_num_samples;
			ImGui::SliderInt("###3", &sample_num_per_pixel, 1, 16, "sample num per pixel = %d");
			layer.sqrt_num_samples = sqrt(sample_num_per_pixel);
		}

		ImGui::Checkbox("Cut off high varient result", &layer.cut_off_high_variance_result);

		ImGui::End();
	}

	void DrawSettings() {
		ImGui::Begin("Settings", &show_setting_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 160));
		
		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
		float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
		next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;

		bool auto_resize_b = auto_resize != 2;
		ImGui::Checkbox("Free size window", &auto_resize_b);
		if (auto_resize_b) auto_resize = 0;
		else auto_resize = 2;

		static int style_idx = 0;
		if (ImGui::Combo("Style###StyleSelector", &style_idx, "Dark\0Light\0"))
		{
			switch (style_idx)
			{
			case 0: ImGui::StyleColorsDark(); break;
			case 1: ImGui::StyleColorsLight(); break;
			}
		}

		ImGuiIO& io = ImGui::GetIO();
		ImGui::SliderFloat("Scale", &io.FontGlobalScale, 0.5, 2, "%.1f");


		ImGui::End();
	}

	void DrawOutput() {
		ImGui::Begin("Output", &show_output_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 100));

		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
		next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;

		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

		static string name = "VRenderer Output";
		ImGui::InputText("Output name", &name[0], name.size()+1, ImGuiInputTextFlags_::ImGuiInputTextFlags_NoHorizontalScroll | ImGuiInputTextFlags_::ImGuiInputTextFlags_AutoSelectAll);

		if (ImGui::Button("Save")) {
			thread([&]() { layer.SaveResultToFile(name); }).detach();
		}

		ImGui::End();
	}

	void DrawShader() {
		ImGui::Begin("Shaders", &show_shader_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

		ImGui::SetWindowPos(next_Window_pos);
		auto& shader_table = VShader::GetAllShaders();
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 80 * (shader_table.size() > 5 ? 5 : (shader_table.size() < 1 ? 1 : shader_table.size()))));

		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		float tmp_x = ImGui::GetWindowWidth() + next_Window_pos.x;
		next_Window_pos_2.x = next_Window_pos_2.x > tmp_x ? next_Window_pos_2.x : tmp_x;

		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
		
		int index = 0;
		for each (auto shader in shader_table)
		{
			ImGui::Text(("Shader: " + shader.first).c_str());
			if (ImGui::Button(("Open Shader###shader_open_" + to_string(index)).c_str())) {
				OpenShaderFile(shader.first);
			}
			ImGui::SameLine();
			if (ImGui::Button(("Reload Shader###shader_reload_" + to_string(index)).c_str())) {
				layer.Lock();
				VShader::Reload(shader.first);
				layer.MarkDirty();
				layer.Unlock();
			}
			ImGui::Separator();
			index++;
		}

		ImGui::End();
	}

	void DrawMaterial() {
		ImGui::Begin("Material", &show_material_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

		ImGui::SetWindowPos(next_Window_pos_2);

		auto& mat_table = VMaterial::GetAllMaterials();

		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(400, 200 * (mat_table.size() > 3 ? 3 : (mat_table.size() < 1 ? 1 : mat_table.size()))));

		next_Window_pos_2 = next_Window_pos_2 + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

		int index = 0;
		for each (auto mat in mat_table)
		{
			ImGui::Text(mat.first.c_str());
			if (ImGui::Button(("Open File###" + to_string(index)).c_str())) {
				OpenMaterialFile(mat.first);
			}
			ImGui::SameLine();
			if (ImGui::Button(("Reload file###1000" + to_string(index)).c_str())) {
				layer.Lock();
				VMaterial::ReloadMaterial(mat.first);
				layer.Unlock();
			}
			ImGui::SameLine();
			if (ImGui::Button(("Save to file###2000" + to_string(index)).c_str())) {
				mat.second->SaveToFile();
			}

			auto propertie = mat.second->GetAllProperties();
			int index2 = 0;
			for each (auto prop in propertie)
			{
				if (prop.second.Type() == "int") {
					if (ImGui::InputInt((prop.first + "###Mat_prop" + to_string(index*1000+index2)).c_str(), prop.second.GetData<int>())) {
						mat.second->MarkDirty();
					}
				}
				else if (prop.second.Type() == "string") {
					if (prop.first == "Shader") {
						LeftLabelText(prop.first + ":  ", prop.second.GetData<string>()->c_str());
					}
					else {
						ImGui::InputText((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<string>());
					}
				}
				else if (prop.second.Type() == "float") {
					if (ImGui::DragFloat((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>(), 0.001, 0, 1)) {
						mat.second->MarkDirty();
					}
				}
				else if (prop.second.Type() == "float2") {
					if (ImGui::DragFloat2((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>(), 0.001, 0, 1)) {
						mat.second->MarkDirty();
					}
				}
				else if (prop.second.Type() == "float3") {
					if (prop.first[0] == '!') {
						if (ImGui::InputFloat3((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
							mat.second->MarkDirty();
						}
					}
					else {
						if (ImGui::ColorEdit3((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
							mat.second->MarkDirty();
						}
					}
				}
				else if (prop.second.Type() == "float4") {
					if (prop.first[0] == '!') {
						if (ImGui::InputFloat4((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
							mat.second->MarkDirty();
						}
					}
					else {
						if (ImGui::ColorEdit4((prop.first + "###Mat_prop" + to_string(index * 1000 + index2)).c_str(), prop.second.GetData<float>())) {
							mat.second->MarkDirty();
						}
					}
				}
				index2++;
			}

			ImGui::Separator();
			index++;
		}

		ImGui::End();
	}

	void DrawScene() {
		ImGui::Begin("Scene", &show_scene_window, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

		ImGui::SetWindowPos(next_Window_pos_2);

		auto& mat_table = VMaterial::GetAllMaterials();

		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(400, 200));

		next_Window_pos_2 = next_Window_pos_2 + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

		static string scene_path = "Default Cornell";
		static FileLoader fl;
		static bool openFile = false;
		ImGui::Text(scene_path.c_str());
		ImGui::SameLine();
		if (ImGui::Button("Open scene")) { openFile = true; };
		if (fl.Draw(needed_size, ".scene", openFile)) {
			scene_path = fl.result;
			VScene::LoadScene(scene_path);
		}
		ImGui::Separator();

		VTransform* root = VScene::root;

		int index = 0;
		if (root != NULL) {
			for each (auto child in root->Childs())
			{
				DisplayNode(index, child);
			}
		}


		ImGui::End();
	}

	void DisplayNode(int& gui_id, VTransform* trans) {
		string name = trans->Object()->name;
		name = name + "###" + to_string(gui_id);
		if (ImGui::TreeNode(name.c_str())) {
			bool changed = false;
			changed |= ImGui::DragFloat3(("Position###pos_input" + to_string(gui_id)).c_str(), trans->Position<float>());
			changed |= ImGui::DragFloat3(("Rotation###rot_input" + to_string(gui_id)).c_str(), trans->Rotation<float>());
			changed |= ImGui::DragFloat3(("Scale###sca_input" + to_string(gui_id)).c_str(), trans->Scale<float>(),0.01, 0);
			gui_id++;
			if (changed) trans->MarkDirty();

			for each (auto child in trans->Childs())
			{
				DisplayNode(gui_id, child);
			}
			ImGui::TreePop();
		}
		else {
			gui_id++;
		}
	}


public:
	VGUI() :layer(OptiXLayer::Instance()) {
		auto& style = ImGui::GetStyle();
		style.FrameRounding = 12.f;
		style.GrabRounding = 12.f;
	}


	void OnDrawGUI() {

		//Main Menu
		DrawMainMenu();

		// Console
		if (show_console_window) DrawConsole();

		// Settings
		if (show_setting_window) DrawSettings();

		// Output
		if (show_output_window) DrawOutput();

		// Shader
		if (show_shader_window) DrawShader();

		// Scene
		if (show_scene_window) DrawScene();

		// Material
		if (show_material_window) DrawMaterial();
	}



};



































#include "Imgui/imgui.h"
#include "Imgui/imgui_impl_dx9.h"
#include "Imgui/imgui_impl_win32.h"
#include <d3d9.h>
#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#include <tchar.h>

// Data
static LPDIRECT3D9              g_pD3D = NULL;
static LPDIRECT3DDEVICE9        g_pd3dDevice = NULL;
static D3DPRESENT_PARAMETERS    g_d3dpp = {};

// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void ResetDevice();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Main code
int Main_Loop(int, char**)
{
	// Create application window
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("VRenderer Controller"), NULL };
	::RegisterClassEx(&wc);
	HWND hwnd = ::CreateWindowEx(/*WS_EX_TOPMOST | */WS_EX_TRANSPARENT | WS_EX_LAYERED, wc.lpszClassName, _T("Controller"), WS_OVERLAPPEDWINDOW^WS_THICKFRAME, 612, 100, 512, 1024, NULL, NULL, wc.hInstance, NULL);
	SetLayeredWindowAttributes(hwnd, 0, 1.0f, LWA_ALPHA);
	SetLayeredWindowAttributes(hwnd, 0, RGB(0, 0, 0), LWA_COLORKEY);
	LONG_PTR Style = ::GetWindowLongPtr(hwnd, GWL_STYLE);
	Style = Style & ~WS_CAPTION &~WS_SYSMENU &~WS_SIZEBOX;
	::SetWindowLongPtr(hwnd, GWL_STYLE, Style);
	DWORD dwExStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
	dwExStyle &= ~(WS_VISIBLE);
	dwExStyle |= WS_EX_TOOLWINDOW;
	dwExStyle &= ~(WS_EX_APPWINDOW);
	SetWindowLong(hwnd, GWL_EXSTYLE, dwExStyle);
	ShowWindow(hwnd, SW_SHOW);
	ShowWindow(hwnd, SW_HIDE);
	UpdateWindow(hwnd);

	DX_Window = hwnd;

	// Initialize Direct3D
	if (!CreateDeviceD3D(hwnd))
	{
		CleanupDeviceD3D();
		::UnregisterClass(wc.lpszClassName, wc.hInstance);
		return 1;
	}

	// Show the window
	::ShowWindow(hwnd, SW_SHOWDEFAULT);
	::UpdateWindow(hwnd);

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplWin32_Init(hwnd);
	ImGui_ImplDX9_Init(g_pd3dDevice);

	VGUI gui;

	SetWindowLong(hwnd, GWL_EXSTYLE, (GetWindowLong(hwnd, GWL_EXSTYLE) & ~WS_EX_TRANSPARENT) | WS_EX_LAYERED);

	// Main loop
	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	while (msg.message != WM_QUIT)
	{
		// Poll and handle messages (inputs, window resize, etc.)
		// You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
		// - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
		// - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
		// Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
		if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			::TranslateMessage(&msg);
			::DispatchMessage(&msg);
			continue;
		}

		// Start the Dear ImGui frame
		ImGui_ImplDX9_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();

		gui.OnDrawGUI();

		// Rendering
		ImGui::EndFrame();

		g_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
		if (g_pd3dDevice->BeginScene() >= 0)
		{
			ImGui::Render();
			ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
			g_pd3dDevice->EndScene();
		}
		
		HRESULT result = g_pd3dDevice->Present(NULL, NULL, NULL, NULL);

		//Handle loss of D3D9 device
		if (result == D3DERR_DEVICELOST && g_pd3dDevice->TestCooperativeLevel() == D3DERR_DEVICENOTRESET)
			ResetDevice();

		RECT rect;
		GetWindowRect(GL_Window, &rect);
		MoveWindow(DX_Window, rect.right, rect.top, gui.needed_size.x, gui.needed_size.y, TRUE);

		static bool active = false;
		if (GetForegroundWindow() == GL_Window) {
			if (active == false) {
				active = true;
			}
			auto window = GetNextWindow(GetTopWindow(0), GW_HWNDNEXT);
			RECT rect;
			GetWindowRect(DX_Window, &rect);
			SetWindowPos(DX_Window, GL_Window, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_NOACTIVATE);
		}
		else
		{
			active = false;
		}
	}

	ImGui_ImplDX9_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();

	CleanupDeviceD3D();
	::DestroyWindow(hwnd);
	::UnregisterClass(wc.lpszClassName, wc.hInstance);

	return 0;
}

// Helper functions

bool CreateDeviceD3D(HWND hWnd)
{
	if ((g_pD3D = Direct3DCreate9(D3D_SDK_VERSION)) == NULL)
		return false;

	// Create the D3DDevice
	ZeroMemory(&g_d3dpp, sizeof(g_d3dpp));
	g_d3dpp.Windowed = TRUE;
	g_d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
	g_d3dpp.hDeviceWindow = hWnd;
	g_d3dpp.MultiSampleQuality = D3DMULTISAMPLE_NONE;
	g_d3dpp.BackBufferFormat = D3DFMT_A8R8G8B8;
	g_d3dpp.BackBufferWidth = 512;
	g_d3dpp.BackBufferHeight = 1024;
	g_d3dpp.EnableAutoDepthStencil = TRUE;
	g_d3dpp.AutoDepthStencilFormat = D3DFMT_D16;
	g_d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_ONE;           // Present with vsync
	//g_d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;   // Present without vsync, maximum unthrottled framerate
	if (g_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &g_d3dpp, &g_pd3dDevice) < 0)
		return false;

	return true;
}

void CleanupDeviceD3D()
{
	if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
	if (g_pD3D) { g_pD3D->Release(); g_pD3D = NULL; }
}

void ResetDevice()
{
	ImGui_ImplDX9_InvalidateDeviceObjects();
	HRESULT hr = g_pd3dDevice->Reset(&g_d3dpp);
	if (hr == D3DERR_INVALIDCALL)
		IM_ASSERT(0);
	ImGui_ImplDX9_CreateDeviceObjects();
}

// Win32 message handler
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
		return true;

	if (msg == WM_SETFOCUS)
	{
		auto window = GetNextWindow(GetTopWindow(0), GW_HWNDNEXT);
		RECT rect;
		GetWindowRect(GL_Window, &rect);
		SetWindowPos(GL_Window, DX_Window, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, SWP_NOACTIVATE);

	}

	switch (msg)
	{
	case WM_SIZE:
		if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
		{
			g_d3dpp.BackBufferWidth = LOWORD(lParam);
			g_d3dpp.BackBufferHeight = HIWORD(lParam);
			ResetDevice();
		}
		return 0;
	case WM_SYSCOMMAND:
		if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
			return 0;
		break;
	case WM_DESTROY:
		::PostQuitMessage(0);
		return 0;
	}
	return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

#endif // !GUI_HPP_