#pragma once

#include "VRender/VRender.hpp"
#include <atomic>

namespace VRender {

	class VScene {
		static string name;

		static vector<string> split(const string& str, const string& delim);

		static void ProcessObject(const string& data);

		static void ProcessLight(const string& data);

		static void ProcessCamera(const string& data);

		static void ProcessLine(const string& line) {
			auto parts = split(line, ":");

			string type = parts[0];

			if (type == "Name") {
				name = parts[1];
			}
			else if (type == "Object") {
				ProcessObject(parts[1]);
			}
			else if (type == "Light") {
				ProcessLight(parts[1]);
			}
			else if (type == "Camera") {
				ProcessCamera(parts[1]);
			}
		}




	public:
		static atomic<float> percentage;

		static string SceneName() { return name; }

		static bool LoadScene(string path) {
			
			VRender::VRenderer::Instance().SetSelectedObject(-1);
			VObjectManager::RemoveAll();
			VLightManager::RemoveAll();
			prime::PrimeComponentManager::Clear();
			
			int line = 1;
			fstream fin;
			try
			{
				sutil::setProjectDir(path.c_str());

				fin.open((string(sutil::projectDir()) + "/Scene.txt").c_str(), ios::in);
				if (!fin) throw Exception("Scene file not exist.");

				vector<string> lines;
				string tmp;
				while (getline(fin, tmp)) {
					lines.push_back(tmp);
				}
				if (lines.empty()) throw Exception("Empty scene file!");
				for each (string str in lines) {
					ProcessLine(str);
					line++;
					percentage = (float)line / lines.size();
				}

				fin.close();
				return true;
			}
			catch (const Exception& e)
			{
				VObjectManager::RemoveAll();
				VLightManager::RemoveAll();
				cout << "Load scene faild:" << endl << "Line " << line << ":" << e.getErrorString() << endl;
				fin.close();
				return false;
			}
		}

		static void Cornell() {
			LoadScene(string(sutil::samplesDir()) + "/Cornell");
		}
	};



}