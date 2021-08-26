#pragma once
#pragma once 
#include "GL/glew.h"
#include "Texture.h"
class BufferExchanger
{
public:
	GLuint pbo = 0;
	BufferExchanger();

	void ReadFromTexture(Texture* tex);
	void WriteToTexture(Texture* tex);

};

