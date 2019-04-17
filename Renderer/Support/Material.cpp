#include "Material.h"


VMaterialManager & VMaterialManager::Instance() {
	static VMaterialManager& ins = VMaterialManager();
	return ins;
}
