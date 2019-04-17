#ifndef GUI_HPP_
#define GUI_HPP_

#include <GlutEvent.h>

#include <dwmapi.h>
#include "Imgui/imgui.h"
#include "Imgui/imgui_impl_win32.h"
#include "Imgui/imgui_impl_dx12.h"
#include <d3d12.h>
#include <dxgi1_4.h>
#include <tchar.h>
#include <time.h>

class VGUI {
	OptiXLayer& layer;

	bool show_setting_window = false;
	bool show_console_window = false;
	bool show_output_window = false;

	ImVec2 next_Window_pos;

	int auto_resize = 2;

public:
	ImVec4 back_ground_color = ImVec4(1.f, 0.55f, 1.f, 1.f);
	ImVec2 needed_size;


private:
	void DrawMainMenu() {
		static float f = 0.0f;
		static int counter = 0;

		ImGui::Begin("Main Menu", 0, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | 
										ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
										ImGuiWindowFlags_::ImGuiWindowFlags_NoResize);
		ImGui::SetWindowPos(ImVec2(0, 0));
		ImGui::SetWindowSize(ImVec2(250, 180));
		next_Window_pos = ImGui::GetWindowPos() + ImVec2(ImGui::GetWindowWidth(), 0);
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
		ImGui::Separator();

		if (ImGui::Button("Focuse Renderer window")) {
			SetForegroundWindow(GL_Window);
			SetFocus(GL_Window);
		}

		ImGui::End();
	}

	void DrawConsole() {
		ImGui::Begin("Console", 0, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300,240));

		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));
		
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
		}

		ImGui::Separator();

		static bool post = false;
		if (layer.resultType != OptiXLayer::ResultBufferType::origial) {
			if (post == false) layer.RebuildCommandList(true);
			ImGui::SliderFloatLableOnLeft("Exposure", &layer.exposure, 0, 100, "%.4f", 3);
			post = true;
			ImGui::Separator();
		}
		else {
			if (post == true) layer.RebuildCommandList(false);
			post = false;
		}

		if (ImGui::SliderFloatLableOnLeft("Diffuse strength", &layer.diffuse_strength, 0, 10, "%.2f")) layer.MaskDirty();

		ImGui::End();
	}

	void DrawSettings() {
		ImGui::Begin("Settings", 0, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);
		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 120));
		
		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

		bool auto_resize_b = auto_resize != 2;
		ImGui::Checkbox("Auto resize", &auto_resize_b);
		if (auto_resize_b) auto_resize = 0;
		else auto_resize = 2;

		ImGui::End();
	}

	void DrawOutput() {
		ImGui::Begin("Output", 0, ImGuiWindowFlags_::ImGuiWindowFlags_NoMove | ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse | auto_resize);

		ImGui::SetWindowPos(next_Window_pos);
		if (auto_resize == 2) ImGui::SetWindowSize(ImVec2(300, 130));

		next_Window_pos = next_Window_pos + ImVec2(0, ImGui::GetWindowHeight());
		auto tmp = ImGui::GetWindowPos() + ImGui::GetWindowSize();
		needed_size = ImVec2(max(needed_size.x, tmp.x), max(needed_size.y, tmp.y));

		string name = "VRenderer Output";
		ImGui::InputText("Output name: ", &name[0], name.size()+1, ImGuiInputTextFlags_::ImGuiInputTextFlags_NoHorizontalScroll);

		if (ImGui::Button("Save")) {
			thread([&]() {layer.SaveResultToFile("name"); }).detach();			
		}
		
		ImGui::End();
	}

public:
	VGUI() :layer(OptiXLayer::Instance()) {}


	void OnDrawGUI() {

		//Main Menu
		DrawMainMenu();

		// Settings
		if (show_console_window) DrawConsole();

		if (show_setting_window) DrawSettings();

		if (show_output_window) DrawOutput();
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