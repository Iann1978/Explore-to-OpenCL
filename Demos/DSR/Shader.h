#pragma once
#include "GL/glew.h"
#include <map>
class Shader
{
public:
	char name[512];
	GLuint program;
public:
	Shader(const char* name,
		const char* vertexShaderFile,
		const char* fragmentShaderFile,
		const char* geometryShaderFile = nullptr,
		const char* vertexIncludeFile = nullptr,
		const char* fragmentIncludeFile = nullptr,
		const char* geometryIncludeFile = nullptr
	);
	Shader(const char* name);
	~Shader();

	void Load(const char* incfile, const char* vsfile, const char* fsfile,
		const char* gsfile = nullptr,
		const char* tcsfile = nullptr,
		const char* tesfile = nullptr);
	void Load2(const char* vsfile, const char* fsfile,
		const char* gsfile = nullptr,
		const char* tcsfile = nullptr,
		const char* tesfile = nullptr);
	
	GLuint GetLocation(const char *valuename);
public:
	static Shader *Find(const char *name);
	//static glm::vec2 modelPosition2D;
	static float modelRotation2D;
};

