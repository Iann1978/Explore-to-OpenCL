#pragma once
#pragma once 
#include "GL/glew.h"
#include "Texture.h"
class BufferExchanger
{
public:
	GLuint pbo = 0;
	Texture* tex;
	BufferExchanger(Texture* tex);

	void ReadFromTexture(Texture* tex);
	void WriteToTexture(Texture* tex);

};

