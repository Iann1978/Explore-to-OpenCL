#pragma once
#include "Shader.h"
#include "Material.h"
#include "Mesh.h"
#include "Texture.h"
#include "BufferExchanger.h"

class RenderEngine
{
public:
	Shader* shader = nullptr;
	Material* material = nullptr;
	Mesh* mesh = nullptr;
	Texture* texture = nullptr;
	BufferExchanger* exchanger;
public:
	RenderEngine(int* argc, char** argv);
	void run();

private:
	bool initGL(int* argc, char** argv);
public:
	void display1();

};

