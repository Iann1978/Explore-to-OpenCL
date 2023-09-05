#include "GL/glew.h"
#include <GL/glut.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
static int window_width = 512;
static int window_height = 512;
static GLuint  vbo = 0;
cudaGraphicsResource* cudavbo = nullptr;
bool moveleft(float* vert);
void moveleft();

void display() {
	glClearColor(1, 0, 0,0);
	glClear(GL_COLOR_BUFFER_BIT);
    glColor3f(0, 1, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(
        0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glDrawArrays(GL_TRIANGLES, 0, 9);
	glutSwapBuffers();
}
void keyboard(unsigned char key, int x, int y) {
    switch (key) {
	case 27:
		exit(0);
		break;
    case 'a':
        moveleft();
        printf("a\n");
        break;
	}
}
bool initGL(int* argc, char** argv) {
    std::cout << "initGL" << std::endl;
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutIdleFunc(display);
    glutKeyboardFunc(keyboard);
    ////glutMotionFunc(motion);
    ////glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    //// initialize necessary OpenGL extensions
    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // create vbo
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float data[] = { 0,0,0,1,0,0,0,1,0 };
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_DYNAMIC_DRAW);

    return true;

}
void moveleft() {
    cudaError_t cudaStatus;
    cudaStatus = cudaGraphicsMapResources(1, &cudavbo, 0);
    size_t num_bytes;
    float* devvert = 0;
    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&devvert, &num_bytes, cudavbo);
    moveleft(devvert);
    cudaStatus = cudaGraphicsUnmapResources(1, &cudavbo, 0);
}

void initCuda() {
    std::cout << "initCuda" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaGraphicsGLRegisterBuffer(&cudavbo, vbo, cudaGraphicsMapFlagsWriteDiscard);

    moveleft();
    int a = 0;
    int b = 0;



}

int main(int argc, char** argv) {
    initGL(&argc, argv);
    initCuda();
	//glutInit(&argc, argv);
	//glutCreateWindow("OpenGL Window");
	//glutDisplayFunc(display);
	//glewInit();
	glutMainLoop();
	return 0;
}