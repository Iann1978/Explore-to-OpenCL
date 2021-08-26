#pragma once

#include <map>
#include "GL/glew.h"
#include "glm/Glm.hpp"
#include <string>
#include <vector>
#include <functional>

class Shader;
class Texture;
class CubeMap;
class Material
{
public:
	int queue0;
	Shader *shader;

	std::map<std::string, int> intValues;
	std::map<std::string, float> floatValues;
	std::map<std::string, glm::vec2> vec2Values;
	std::map<std::string, glm::vec3> vec3Values;
	std::map<std::string, glm::vec4> vec4Values;
	std::map<std::string, glm::mat4> mat4Values;

	
	std::map<std::string, GLuint> texValues;
	std::map<std::string, Texture*> texValues1;
	//std::map<std::string, TexturePtr> texValues2;
	std::map<std::string, CubeMap*> cubeValues;

	
public:
	//Material(std::string shaderName);
	Material(Shader *shader);
	~Material();


public:
	void SetInt(const char *name, const int value);
	void SetFloat(const char *name, const float value);
	void SetVector(const char* name, const glm::vec2 value);
	void SetVector(const char *name, const glm::vec3 value);
	void SetVector(const char *name, const glm::vec4 value);
	void SetMatrix(const char* name, const glm::mat4 value);
	void SetTexture(const char* name, const GLuint texture);

	void SetTexture(const char* name, Texture* texture);

	void SetCubeMap(const char* name, CubeMap* cubemap);

	Texture* GetTexture(const char* name);

	void Use();

public:
	typedef std::function<void(void)> ConfigStatus;
	ConfigStatus configStatus;
	static void ConfigStatus_Geomtery();
	static void ConfigStatus_Transparent();
	static void ConfigStatus_UI2D();
	static void ConfigStatus_Add();
	static void ConfigStatus_Transparent_SubMask();
	static void ConfigStatus_Mask();


};

