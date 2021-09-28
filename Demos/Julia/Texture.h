#pragma once
#include "GL/glew.h"
#include <string>

class Texture
{
public:
	enum Usage
	{
		BaseTexture = 1,
		HeightMap = 2,
	};

public:
	unsigned int texture;
	int width;
	int height;
	std::string source;
	std::string uuid;

public:
	Texture() : texture(0), width(0), height(0) {};
	Texture(int width, int height, Usage usage=BaseTexture);
	~Texture();

public:
	int CreateBaseTexture();
	int CreateHeightMap();
	int UpdateTexture(unsigned char* rgba, int width, int height);
	int UpdateHeightMapTexture(float* data, int width, int height);

	void Repeat();

public:
	static Texture* LoadTexture(const char* pathname);
	static int LoadTexture(const char* pathname, Texture*);
};
