#pragma once
#include "Shader.h"
#include "Material.h"
#include "Mesh.h"
#include "Texture.h"
#include "BufferExchanger.h"

class GLCudaEngine
{
public:
	Shader* shader = nullptr;
	Material* material = nullptr;
	Mesh* mesh = nullptr;
	Texture* texture = nullptr;
	BufferExchanger* exchanger = nullptr;
public:
	GLCudaEngine();
	void InitGL(int* argc, char** argv);
	bool InitCuda();
	void UpdateCuda();
	void Run();
public:
	void display1();
};

