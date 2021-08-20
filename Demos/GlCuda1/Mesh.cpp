
#include "Mesh.h"

static const GLfloat quad2_vertexBufferData[] = {
	0.0f,0.0f,0.0f,
	1.0f,0.0f, 0.0f,
	1.0f, 1.0f, 0.0f,
	0.0f,1.0f,0.0f,
};

static const GLfloat quad2_uvBufferData[] = {
	0.0f, 0.0f,
	1.0f, 0.0f,
	1.0f, 1.0f,
	0.0f, 1.0f,
};

static const GLfloat quad2_uvBufferData_FlipY[] = {
	0.0f, 1.0f,
	1.0f, 1.0f,
	1.0f, 0.0f,
	0.0f, 0.0f,
};

static const unsigned short quad2_elementBufferData[] = {
	0,1,2,0,2,3
};
Mesh::Mesh()
{
	buildinType = NotBuildinType;
	patchVertices = 3;

	vertexbuffer = 0;
	uvbuffer = 0;
	normalbuffer = 0;
	tangentbuffer = 0;
	colorbuffer = 0;
	elementbuffer = 0;
	elementsize = 0;
}
Mesh::Mesh(Type type, const int vertexBufferLength, const float* vertexBufferData,
	const int uvBufferLength, const float *uvBufferData, 
	const int colorBufferLength, const float *colorBufferData,
	const int elementBufferLength, const unsigned short *elementBufferData)
{
	buildinType = NotBuildinType;
	patchVertices = 3;

	vertexbuffer = 0;
	uvbuffer = 0;
	normalbuffer = 0;
	tangentbuffer = 0;
	colorbuffer = 0;
	elementbuffer = 0;
	elementsize = 0;
	this->type = type;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertexBufferLength * sizeof(float), vertexBufferData, GL_STATIC_DRAW);

	if (uvBufferData)
	{
		glGenBuffers(1, &uvbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glBufferData(GL_ARRAY_BUFFER, uvBufferLength * sizeof(float), uvBufferData, GL_STATIC_DRAW);
	}

	if (colorBufferData)
	{
		glGenBuffers(1, &colorbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, colorBufferLength * sizeof(float), colorBufferData, GL_STATIC_DRAW);
	}
	
	if (elementBufferData)
	{
		glGenBuffers(1, &elementbuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, elementBufferLength * sizeof(unsigned short), elementBufferData, GL_STATIC_DRAW);
	}

	elementsize = elementBufferLength;
}



Mesh::~Mesh()
{
	if (vertexbuffer)
	{
		glDeleteBuffers(1, &vertexbuffer);
		vertexbuffer = 0;
	}
	if (uvbuffer)
	{
		glDeleteBuffers(1, &uvbuffer);
		uvbuffer = 0;
	}
	if (colorbuffer)
	{
		glDeleteBuffers(1, &colorbuffer);
		colorbuffer = 0;
	}
	if (elementbuffer)
	{
		glDeleteBuffers(1, &elementbuffer);
		elementbuffer = 0;
	}
}

void Mesh::UpdateVertexBuffer(const int vertexBufferLength, const float* vertexBufferData)
{
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertexBufferLength * sizeof(float), vertexBufferData, GL_STATIC_DRAW);
}



Mesh *Mesh::CreateQuad(glm::vec4 rect)
{
	float x = rect.x;
	float y = rect.y;
	float w = rect.z;
	float h = rect.w;

	float vertexBufferData[] = {
		x,		y,		0.0f,
		x + w,	y,		0.0f,
		x + w,	y + h,	0.0f,
		x,		y + h,	0.0f,
	};

	return new Mesh(Triangles, 12, vertexBufferData,
		8, quad2_uvBufferData,
		0, nullptr,
		6, quad2_elementBufferData);
}



Mesh* Mesh::CreatePlane(glm::vec4 rect, int row, int col)
{
	MeshBuilder meshBuilder;
	meshBuilder.type = Mesh::Type::Triangles;
	float w = rect.z;
	float h = rect.w;
	float dx = w / (col - 1);
	float dz = h / (row - 1);

	for (int r = 0; r < row; r++)
	{
		for (int c = 0; c < col; c++)
		{
			float x = rect.x + c * dx;
			float y = 0;
			float z = rect.y + r * dz;
			glm::vec3 v(x, y, z);
			meshBuilder.vertices.push_back(v);
			meshBuilder.normal.push_back(glm::vec3(0, 1, 0));
			meshBuilder.tangent.push_back(glm::vec3(1, 0, 0));
			meshBuilder.uv.push_back(glm::vec2(1.0f*c/(col-1),1.0f*r/(row-1)));
		}
	}

	for (int r = 0; r < row-1; r++)
	{
		for (int c = 0; c < col-1; c++)
		{
			
			short i0 = r * col + c;
			short i1 = r * col + (c+1);
			short i2 = (r+1) * col + (c + 1);
			short i3 = (r + 1) * col + c;
			meshBuilder.indices.push_back(i0);
			meshBuilder.indices.push_back(i2);
			meshBuilder.indices.push_back(i1);
			
			meshBuilder.indices.push_back(i0);
			meshBuilder.indices.push_back(i3);
			meshBuilder.indices.push_back(i2);
			
		}
	}
	
	
	Mesh* mesh = meshBuilder.build(Mesh::Type::Triangles);
	mesh->buildinType = Mesh::BuildinType::Plane;
	return mesh;
}
Mesh* Mesh::CreateQuadFlipY(glm::vec4 rect)
{
	float x = rect.x;
	float y = rect.y;
	float w = rect.z;
	float h = rect.w;

	float vertexBufferData[] = {
		x,		y,		0.0f,
		x + w,	y,		0.0f,
		x + w,	y + h,	0.0f,
		x,		y + h,	0.0f,
	};

	return new Mesh(Triangles, 12, vertexBufferData,
		8, quad2_uvBufferData_FlipY,
		0, nullptr,
		6, quad2_elementBufferData);
}



Mesh* Mesh::CreateBoundingBox(glm::vec3 min, glm::vec3 max)
{
	glm::vec3 v0(min.x, min.y, min.z);
	glm::vec3 v1(max.x, min.y, min.z);
	glm::vec3 v2(max.x, max.y, min.z);
	glm::vec3 v3(min.x, max.y, min.z);

	glm::vec3 v4(min.x, min.y, max.z);
	glm::vec3 v5(max.x, min.y, max.z);
	glm::vec3 v6(max.x, max.y, max.z);
	glm::vec3 v7(min.x, max.y, max.z);

	MeshBuilder meshBuilder;
	meshBuilder.type = Mesh::Type::Lines;

	meshBuilder.vertices.push_back(v0);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v1);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v2);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v3);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v4);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v5);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v6);
	//meshBuilder.color.push_back(Color::white);
	meshBuilder.vertices.push_back(v7);
	//meshBuilder.color.push_back(Color::white);




	meshBuilder.indices.push_back(0);
	meshBuilder.indices.push_back(1);
	meshBuilder.indices.push_back(1);
	meshBuilder.indices.push_back(2);
	meshBuilder.indices.push_back(2);
	meshBuilder.indices.push_back(3);
	meshBuilder.indices.push_back(3);
	meshBuilder.indices.push_back(0);

	meshBuilder.indices.push_back(0 + 4);
	meshBuilder.indices.push_back(1 + 4);
	meshBuilder.indices.push_back(1 + 4);
	meshBuilder.indices.push_back(2 + 4);
	meshBuilder.indices.push_back(2 + 4);
	meshBuilder.indices.push_back(3 + 4);
	meshBuilder.indices.push_back(3 + 4);
	meshBuilder.indices.push_back(0 + 4);



	meshBuilder.indices.push_back(0);
	meshBuilder.indices.push_back(0 + 4);
	meshBuilder.indices.push_back(1);
	meshBuilder.indices.push_back(1 + 4);
	meshBuilder.indices.push_back(2);
	meshBuilder.indices.push_back(2 + 4);
	meshBuilder.indices.push_back(3);
	meshBuilder.indices.push_back(3 + 4);

	return meshBuilder.build(Mesh::Type::Lines);


}
void RenderTrianglesMesh(Mesh *mesh)
{
	
	if (mesh->type != Mesh::Type::Triangles)
	{
		printf("Error in RenderMeshTrianglesMesh");
		return;
	}

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	if (mesh->uvbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);

		//glClientActiveTextureARB(GL_TEXTURE0_ARB);
		//glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		//glTexCoordPointer(sizeof(quad2_uvBufferData), GL_FLOAT, 0, quad2_uvBufferData);

		//glTexCoordPointer(...)

	}


	if (mesh->colorbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(2);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->colorbuffer);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	if (mesh->normalbuffer)
	{
		glEnableVertexAttribArray(3);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->normalbuffer);
		glVertexAttribPointer(
			3,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	if (mesh->tangentbuffer)
	{
		glEnableVertexAttribArray(4);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->tangentbuffer);
		glVertexAttribPointer(
			4,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->elementbuffer);
	
	// Draw the triangles !
	glDrawElements(
		GL_TRIANGLES,      // mode
		mesh->elementsize,    // count
		GL_UNSIGNED_SHORT, // type
		(void*)0           // element array buffer offset
	);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
}


void RenderPatchesMesh(Mesh* mesh)
{

	if (mesh->type != Mesh::Type::Patches)
	{
		printf("Error in RenderPatchesMesh");
		return;
	}

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	if (mesh->uvbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}


	if (mesh->colorbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(2);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->colorbuffer);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	if (mesh->normalbuffer)
	{
		glEnableVertexAttribArray(3);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->normalbuffer);
		glVertexAttribPointer(
			3,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	if (mesh->tangentbuffer)
	{
		glEnableVertexAttribArray(4);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->tangentbuffer);
		glVertexAttribPointer(
			4,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			3,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->elementbuffer);
	glPatchParameteri(GL_PATCH_VERTICES, mesh->patchVertices);
	// Draw the triangles !
	glDrawElements(
		GL_PATCHES,      // mode
		mesh->elementsize,    // count
		GL_UNSIGNED_SHORT, // type
		(void*)0           // element array buffer offset
	);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
}

void RenderPointsMesh(Mesh *mesh)
{
	if (mesh->type != Mesh::Type::Points)
	{
		printf("Error in RenderMeshPointsMesh");
		return;
	}

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	if (mesh->uvbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}


	if (mesh->colorbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(2);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->colorbuffer);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	glDrawArrays(GL_POINTS, 0, 1);
	
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
}

void RenderLineStripMesh(Mesh *mesh)
{
	if (mesh->type != Mesh::Type::LineStrip)
	{
		printf("Error in RenderLineStripMesh");
		return;
	}

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	if (mesh->uvbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(1);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->uvbuffer);
		glVertexAttribPointer(
			1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			2,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}


	if (mesh->colorbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(2);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->colorbuffer);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	glDrawArrays(GL_LINE_STRIP, 0, mesh->elementsize); // 12*3 indices starting at 0 -> 12 triangles

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
}

void RenderLinesMesh(Mesh* mesh)
{
	if (mesh->type != Mesh::Type::Lines)
	{
		printf("Error in RenderLineStripMesh");
		return;
	}

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	//if (mesh->uvbuffer)
	//{
	//	// 2nd attribute buffer : UVs
	//	glEnableVertexAttribArray(1);
	//	//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
	//	glBindBuffer(GL_ARRAY_BUFFER, mesh->uvbuffer);
	//	glVertexAttribPointer(
	//		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
	//		2,                                // size : U+V => 2
	//		GL_FLOAT,                         // type
	//		GL_FALSE,                         // normalized?
	//		0,                                // stride
	//		(void*)0                          // array buffer offset
	//	);
	//}


	if (mesh->colorbuffer)
	{
		// 2nd attribute buffer : UVs
		glEnableVertexAttribArray(2);
		//glBindBuffer(GL_ARRAY_BUFFER, uvbuffer_image);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->colorbuffer);
		glVertexAttribPointer(
			2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
			4,                                // size : U+V => 2
			GL_FLOAT,                         // type
			GL_FALSE,                         // normalized?
			0,                                // stride
			(void*)0                          // array buffer offset
		);
	}

	//glDrawArrays(GL_LINES, 0, mesh->elementsize); // 12*3 indices starting at 0 -> 12 triangles
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->elementbuffer);
	glDrawElements(
		GL_LINES,      // mode
		mesh->elementsize,    // count
		GL_UNSIGNED_SHORT, // type
		(void*)0           // element array buffer offset
	);


	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
}
void Mesh::RenderMesh(Mesh *mesh)
{
	if (!mesh)
		return;
	if (mesh->type == Type::Triangles)
		RenderTrianglesMesh(mesh);
	else if (mesh->type == Type::Points)
		RenderPointsMesh(mesh);
	else if (mesh->type == Type::LineStrip)
		RenderLineStripMesh(mesh);
	else if (mesh->type == Type::Lines)
		RenderLinesMesh(mesh);
	else if (mesh->type == Type::Patches)
		RenderPatchesMesh(mesh);
	else
		printf("Error in Mesh::RenderMesh().");
}


Mesh* MeshBuilder::build(Mesh::Type type)
{
	this->type = type;
	Mesh* mesh = new Mesh();

	mesh->type = type;
	glGenBuffers(1, &mesh->vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * 3 * sizeof(float), vertices.data(), GL_STATIC_DRAW);

	if (uv.size())
	{
		glGenBuffers(1, &mesh->uvbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->uvbuffer);
		glBufferData(GL_ARRAY_BUFFER, uv.size() * 2 * sizeof(float), uv.data(), GL_STATIC_DRAW);
	}

	if (normal.size())
	{
		glGenBuffers(1, &mesh->normalbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->normalbuffer);
		glBufferData(GL_ARRAY_BUFFER, normal.size() * 3 * sizeof(float), normal.data(), GL_STATIC_DRAW);	
	}

	if (tangent.size())
	{
		glGenBuffers(1, &mesh->tangentbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->tangentbuffer);
		glBufferData(GL_ARRAY_BUFFER, tangent.size() * 3 * sizeof(float), tangent.data(), GL_STATIC_DRAW);
	}

	if (color.size())
	{
		glGenBuffers(1, &mesh->colorbuffer);
		glBindBuffer(GL_ARRAY_BUFFER, mesh->colorbuffer);
		glBufferData(GL_ARRAY_BUFFER, color.size() * 4 * sizeof(float), color.data(), GL_STATIC_DRAW);
	}

	if (indices.size())
	{
		glGenBuffers(1, &mesh->elementbuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->elementbuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), indices.data(), GL_STATIC_DRAW);
	}


	mesh->elementsize = (GLuint)indices.size();
	if (Mesh::LineStrip == type)
	{
		mesh->elementsize = vertices.size();
	}
	
	return mesh;
}
