#pragma once
#include "GL/glew.h"
#include "glm/Glm.hpp"
#include <string>
#include <vector>


class Mesh
{
public:
	enum Type
	{
		Triangles,
		LineStrip,
		Points,
		Lines,
		Patches,
	};
	Type type;
	enum BuildinType
	{
		NotBuildinType = 0,
		Plane = 1,
		Sphere = 2,
		Column = 3,
	};
	BuildinType buildinType;
	int patchVertices;
public:
	GLuint vertexbuffer;
	GLuint uvbuffer;
	GLuint normalbuffer;
	GLuint tangentbuffer;
	GLuint colorbuffer;
	GLuint elementbuffer;
	GLuint elementsize;
	std::string source;
public:
	//Mesh(const Mesh&) = delete;
	Mesh();
	Mesh(Type type, const int vertexBufferLength, const float* vertexBufferData,
		const int uvBufferLength, const float *uvBufferData, 
		const int colorBufferLength, const float *colorBufferData,
		const int elementBufferLength, const unsigned short *elementBufferData);
	~Mesh();

public:
	void UpdateVertexBuffer(const int vertexBufferLength, const float* vertexBufferData);
public:
	static Mesh *quad2;//rect(0,0,1,1)

	static Mesh* CreateQuad(glm::vec4 rect);
	/*!
	@brief Create a horizontal plane. 
	@param[in] rect rect.x: minimaze x of plane,
					rect.y: minimaze y of plane,
					rect.z: width of plane,
					rect.w: heigh of plane.
	@param[in] row: the vertex count along x axis.	
	@param[in] col: the vertex count along z axis.
	*/
	static Mesh* CreatePlane(glm::vec4 rect = glm::vec4(-5,-5,10,10), int row=11, int col=11);
	static Mesh* CreateQuadFlipY(glm::vec4 rect);
	static Mesh* CreateBoundingBox(glm::vec3 min, glm::vec3 max);

	static void RenderMesh(Mesh *mesh);
};



class MeshBuilder
{
public:
	Mesh::Type type;
	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uv;
	std::vector<glm::vec3> normal;
	std::vector<glm::vec3> tangent;
	std::vector<glm::vec4> color;
	std::vector<unsigned short> indices;

	Mesh* build(Mesh::Type type);
};