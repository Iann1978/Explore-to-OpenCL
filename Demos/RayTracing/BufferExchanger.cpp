#include "BufferExchanger.h"

BufferExchanger::BufferExchanger(Texture* tex)
	:tex(tex)
{
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	int BUFFER_SIZE = tex->width * tex->height * 4;
	glBufferData(GL_PIXEL_PACK_BUFFER, BUFFER_SIZE, 0, GL_STREAM_COPY);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}


void BufferExchanger::ReadFromTexture(Texture* tex)
{


	GLenum err = glGetError();
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	err = glGetError();
	glActiveTexture(GL_TEXTURE0);
	err = glGetError();
	glBindTexture(GL_TEXTURE_2D, tex->texture);
	err = glGetError();

	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, 0);

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

	//float pix = 0.2f;
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex->width, tex->height, GL_RED, GL_FLOAT, &pix);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex->width, tex->height, GL_RED, GL_FLOAT, 0);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 512, 512, 0, GL_RED, GL_FLOAT, 0);
	err = glGetError();
	glBindTexture(GL_TEXTURE_2D, 0);
	err = glGetError();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	err = glGetError();
}