#pragma once
#include "Texture.h"
class BufferExchanger;
class DSRSimulator
{
public:
	BufferExchanger* changer = nullptr;
	DSRSimulator(BufferExchanger* changer);
	void runOnce();
};

