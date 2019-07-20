#pragma once

#include "VRender/VRender.hpp"


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
		static string SceneName() { return name; }

		static bool LoadScene(string path) {
			
			VObjectManager::RemoveAll();
			VLightManager::RemoveAll();
			
			int line = 1;
			fstream fin;
			try
			{
				sutil::setProjectDir(path.c_str());

				fin.open((string(sutil::projectDir()) + "/Scene.txt").c_str(), ios::in);
				if (!fin) throw Exception("Scene file not exist.");

				string tmp;
				while (getline(fin, tmp)) {
					ProcessLine(tmp);
					line++;
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