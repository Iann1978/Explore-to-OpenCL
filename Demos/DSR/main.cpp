int main1();

#include "RenderEngine.h"
#include "DSRSimulator.h"

int main(int argc, char** argv)
{
	//return main1();
	RenderEngine engine(&argc, argv);
	DSRSimulator(engine.texture);
	engine.run();
}

