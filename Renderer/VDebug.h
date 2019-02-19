#pragma once

#ifdef _DEBUG

#define abandon { throw std::exception("Use undefined function"); }


#else

#define abandon = default



#endif // _DEBUG



