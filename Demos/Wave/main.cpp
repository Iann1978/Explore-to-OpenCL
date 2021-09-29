#include "GLCudaEngine.h"

int main(int argc, char** argv)
{
    GLCudaEngine engine;
    engine.InitGL(&argc, argv);
    engine.InitCuda();
    engine.Run();

	return 0;
}