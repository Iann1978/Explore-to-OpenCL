
#include "Material.h"


#include "Shader.h"
//#include "Texture.h"
//#include "CubeMap.h"



//
//Material::Material(std::string shaderName)
//{	
//	this->shader = Shader::Find(shaderName.c_str());
//}

Material::Material(Shader *shader)
{
	this->shader = shader;
}


Material::~Material()
{
}

void Material::SetInt(const char *name, const int value)
{
	intValues[name] = value;
}
void Material::SetFloat(const char *name, const float value)
{
	floatValues[name] = value;
}
void Material::SetVector(const char* name, const glm::vec2 value)
{
	GLint id = glGetUniformLocation(shader->program, name);
	if (-1 == id)
	{
		printf("Warning: the given uniform name(%s) does not exist in shader(%s).", name, shader->name);
	}
	vec2Values[name] = value;
}
void Material::SetVector(const char *name, const glm::vec3 value)
{
	vec3Values[name] = value;
}
void Material::SetVector(const char *name, const glm::vec4 value)
{
	vec4Values[name] = value;
}
void Material::SetMatrix(const char* name, const glm::mat4 value)
{
	mat4Values[name] = value;
}
void Material::SetTexture(const char* name, const GLuint texture)
{
	texValues[name] = texture;
}
void Material::SetTexture(const char* name, Texture* texture)
{
	//assert(texture);
	if (texture)
	{
		texValues1[name] = texture;
	}
}
//void Material::SetTexture(const char* name, TexturePtr texture)
//{
//	texValues2[name] = texture;
//}
void Material::SetCubeMap(const char* name, CubeMap* cubemap)
{
	if (cubemap)
	{
		cubeValues[name] = cubemap;
	}
}
//Texture* Material::GetTexture(const char* name)
//{
//	if (texValues1.find(name) != texValues1.end())
//	{
//		return texValues1[name];
//	}
//	return nullptr;
//}
//
//TexturePtr Material::GetTexture2(const char* name)
//{
//	if (texValues2.find(name) != texValues2.end())
//	{
//		return texValues2[name];
//	}
//	return nullptr;
//}

void Material::Use()
{
	glUseProgram(shader->program);
	for (std::map<std::string, int>::iterator i = intValues.begin();
		i != intValues.end(); i++)
	{
		
		GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
		glUniform1i(id, i->second);
	}

	for (std::map<std::string, float>::iterator i = floatValues.begin();
		i != floatValues.end(); i++)
	{
		GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
		glUniform1f(id, i->second);
	}

	//for (auto i : vec2Values)
	//{
	//	GLuint id = glGetUniformLocation(shader->program, i.first.c_str());
	//	glUniform2f(id, i.second.x, i.second.y);
	//}

	for (std::map<std::string, glm::vec3>::iterator i = vec3Values.begin();
		i != vec3Values.end(); i++)
	{
		GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
		glUniform3f(id, i->second.x, i->second.y, i->second.z);
	}

	for (std::map<std::string, glm::vec4>::iterator i = vec4Values.begin();
		i != vec4Values.end(); i++)
	{
		GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
		glUniform4f(id, i->second.x, i->second.y, i->second.z, i->second.w);
	}

	for (std::map<std::string, glm::mat4>::iterator i = mat4Values.begin();
		i != mat4Values.end(); i++)
	{
		GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
		glUniformMatrix4fv(id, 1, GL_FALSE, &i->second[0][0]);
	}

	//assert(((bool)(texValues.size()<=2)));
	int texIdx = 0;
	//for (auto i : texValues)
	//{
	//	glActiveTexture(GL_TEXTURE0 + texIdx);
	//	glBindTexture(GL_TEXTURE_2D, i.second);
	//	GLuint id = glGetUniformLocation(shader->program, i.first.c_str());
	//	glUniform1i(id, texIdx);
	//	texIdx++;
	//}

	//for (auto i : texValues1)
	//for (std::map<std::string, Texture*>::iterator i = texValues1.begin();
	//	i != texValues1.end(); i++)
	//{
	//	glActiveTexture(GL_TEXTURE0 + texIdx);
	//	glBindTexture(GL_TEXTURE_2D, i->second->texture);
	//	GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
	//	glUniform1i(id, texIdx);
	//	texIdx++;
	//}

	//for (auto i : texValues2)
	//{
	//	glActiveTexture(GL_TEXTURE0 + texIdx);
	//	glBindTexture(GL_TEXTURE_2D, i.second->texture);
	//	GLuint id = glGetUniformLocation(shader->program, i.first.c_str());
	//	glUniform1i(id, texIdx);
	//	texIdx++;
	//}
	//for (std::map<std::string, CubeMap*>::iterator i = cubeValues.begin();
	//	i != cubeValues.end(); i++)
	//{
	//	glActiveTexture(GL_TEXTURE0 + texIdx);
	//	glBindTexture(GL_TEXTURE_CUBE_MAP, i->second->texture);
	//	GLuint id = glGetUniformLocation(shader->program, i->first.c_str());
	//	glUniform1i(id, texIdx);
	//	texIdx++;
	//}

	glActiveTexture(GL_TEXTURE0);

	if (configStatus)
	{
		configStatus();
	}
}

void Material::ConfigStatus_Geomtery()
{
	glDisable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ZERO);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_STENCIL_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
}

void Material::ConfigStatus_Transparent()
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_STENCIL_TEST);
}

void Material::ConfigStatus_UI2D()
{
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_STENCIL_TEST);
	glDisable(GL_CULL_FACE);
}

void Material::ConfigStatus_Add()
{
	// Depth status
	glDisable(GL_DEPTH_TEST);

	// Stencil status
	glEnable(GL_STENCIL_TEST);
	glStencilMask(0);
	glStencilFunc(GL_EQUAL, 0xFF, ~0);
	//glDisable(GL_STENCIL_TEST); // temp 

	// Blend status
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
}

void Material::ConfigStatus_Transparent_SubMask()
{	
	glDisable(GL_DEPTH_TEST);

	// Config stencil test
	glEnable(GL_STENCIL_TEST);
	glStencilMask(0);
	glStencilFunc(GL_NOTEQUAL, 0xFF, ~0);

	// Config blend 
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
}

void Material::ConfigStatus_Mask()
{

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_NEVER);

	glEnable(GL_STENCIL_TEST);
	glStencilMask(0xFF);
	glStencilFunc(GL_ALWAYS, 0xFF, 0xFF);
	glStencilOp(GL_REPLACE, GL_REPLACE, GL_REPLACE);

}