#include "BufferExchanger.h"

BufferExchanger::BufferExchanger()
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	int BUFFER_SIZE = 512 * 512* 4;
	glBufferData(GL_PIXEL_PACK_BUFFER, BUFFER_SIZE, 0, GL_STREAM_COPY);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

}


void BufferExchanger::ReadFromTexture(Texture* tex)
{
	//if (!buff)
	//{
	//	buff = new float[1024 * 1024];
	//}
	//if (!texture)
	//{
	//	texture = new Texture(512, 512);
	//}
	GLenum err = glGetError();
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	err = glGetError();
	glActiveTexture(GL_TEXTURE0);
	err = glGetError();
	glBindTexture(GL_TEXTURE_2D, tex->texture);
	err = glGetError();
	//glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, NULL);
//	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	//glGetTexImage(GL_TEXTURE_2D, 0,
	//	GL_RGBA,
	//	GL_UNSIGNED_BYTE,
	//	0);

	glGetTexImage(GL_TEXTURE_2D, 0,
		GL_RED,
		GL_FLOAT,
		0);

	err = glGetError();
	glBindTexture(GL_TEXTURE_2D, 0);
	err = glGetError();
	glActiveTexture(GL_TEXTURE0);
	err = glGetError();
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	err = glGetError();

}

void BufferExchanger::WriteToTexture(Texture* tex)
{
	GLenum err = glGetError();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	err = glGetError();
	glActiveTexture(GL_TEXTURE0);
	err = glGetError();
	glBindTexture(GL_TEXTURE_2D, tex->texture);
	err = glGetError();
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 512, 512, GL_RED, GL_FLOAT, NULL);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 512, 0, GL_RED, GL_FLOAT, 0);
	err = glGetError();
	glBindTexture(GL_TEXTURE_2D, 0);
	err = glGetError();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	err = glGetError();
}