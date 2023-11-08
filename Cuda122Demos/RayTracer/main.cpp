// https://raytracing.github.io/books/RayTracingInOneWeekend.html
// https://www.3dgep.com/opengl-interoperability-with-cuda/#The_CUDA_Kernel
#include "GL/glew.h"
#include <GL/glut.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "sphere.h"

static int window_width = 512;
static int window_height = 512;
//static GLuint  vbo = 0;

GLuint shaderProgram = 0;
GLuint texture = 0;
GLuint VAO, VBO[2];
cudaGraphicsResource* cudavbo0 = nullptr;
cudaGraphicsResource* cuda_texture = nullptr;

sphere* world = nullptr;
bool moveleft(float* vert);
void moveleft();
bool render(unsigned char* dev_imgdata, int  img_width, int img_height, int img_channels, sphere* world);
void render();


const char* vertexShaderSource = "#version 330\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec2 uv;\n"
"varying vec2 ouv;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"    ouv = uv;\n"
"}\0";

const char* fragmentShaderSource = "#version 330\n"
"uniform sampler2D tex;\n"
"out vec4 FragColor;\n"
"varying vec2 ouv;\n"
"void main()\n"
"{\n"
"    FragColor = vec4(texture2D(tex,ouv).rgb,1.0);\n"
"}\0";


void display() {

    // Clear the screen
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Use the shader program
    glUseProgram(shaderProgram);

    // set texture to shader
    glActiveTexture(GL_TEXTURE0 + 0);
    glBindTexture(GL_TEXTURE_2D, texture);
    GLuint id = glGetUniformLocation(shaderProgram, "tex");
    glUniform1i(id, 0);

    // Draw the triangle
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);


	//glClearColor(1, 0, 0,0);
	//glClear(GL_COLOR_BUFFER_BIT);
 //   glColor3f(0, 1, 0);
 //   glEnableVertexAttribArray(0);
 //   glBindBuffer(GL_ARRAY_BUFFER, vbo);
 //   glVertexAttribPointer(
 //       0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
 //       3,                  // size
 //       GL_FLOAT,           // type
 //       GL_FALSE,           // normalized?
 //       0,                  // stride
 //       (void*)0            // array buffer offset
 //   );
 //   glDrawArrays(GL_TRIANGLES, 0, 9);
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
    case 'r':
        render();
        printf("r\n");
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

    if (!glewIsSupported("GL_VERSION_3_3 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }


    // Create vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Check vertex shader compilation errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cout << "Vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }

    // Create fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Check fragment shader compilation errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cout << "Fragment shader compilation failed: " << infoLog << std::endl;
        return false;
    }

    // Create shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check shader program linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cout << "Shader program linking failed: " << infoLog << std::endl;
        return false;
    }

    // Delete shaders (already linked to the shader program)
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    // set texture filtering parameters
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // set texture filtering to GL_LINEAR (default filtering method)
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // load image, create texture and generate mipmaps
    GLenum err = glGetError();
    int channels = 3;
    unsigned char* data = new unsigned char[window_width * window_height * channels];
    for (int i = 0; i < window_width * window_height * channels; i++) {
        //    memset(data, 255, width * height * channels);
        data[i] = i % 256;
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    //render(data, width, height, channels);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, window_width, window_height, 0, GL_RGB, GL_UNSIGNED_BYTE, data); // note how we specify the texture's data value to be float
    err = glGetError();
    glBindTexture(GL_TEXTURE_2D, 0);


    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    //// create vbo
    //glGenBuffers(1, &vbo);
    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //float vertices[] = { 0,0,0,1,0,0,0,1,0 };
    //glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);


    // Set up vertex data and buffers
    GLfloat vertices[] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f, 0.5f, 0.0f,

    };

    GLfloat coords[] = {
        0.0f, 1.0f, // lower-left corner  
        1.0f, 1.0f, // lower-right corner
        1.0f, 0.0f,  // top-center corner
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f,
    };


    glGenVertexArrays(1, &VAO);
    glGenBuffers(2, VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(coords), coords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);





    return true;

}
void moveleft() {
    cudaError_t cudaStatus;
    cudaStatus = cudaGraphicsMapResources(1, &cudavbo0, 0);
    size_t num_bytes;
    float* devvert = 0;
    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&devvert, &num_bytes, cudavbo0);
    moveleft(devvert);
    cudaStatus = cudaGraphicsUnmapResources(1, &cudavbo0, 0);
}

void render() {
    cudaError_t cudaStatus = cudaSuccess;
    
    size_t num_bytes = window_width * window_height* 4;
    unsigned char* dev_texture = 0;
    cudaStatus = cudaMalloc((void**)&dev_texture, num_bytes * sizeof(unsigned char));


    render(dev_texture, window_width, window_height, 4, world);



    cudaStatus = cudaGraphicsMapResources(1, &cuda_texture, 0);

    cudaArray* devArray;
    cudaStatus = cudaGraphicsSubResourceGetMappedArray(&devArray, cuda_texture, 0, 0);

    
    //cudaStatus = cudaMemcpyToArray(devArray, 0, 0, (void*)dev_texture, num_bytes, cudaMemcpyDeviceToDevice);
    cudaStatus = cudaMemcpy2DToArray(devArray, 0, 0, (void*)dev_texture, window_width * 4, window_width * 4, window_height, cudaMemcpyDeviceToDevice);


    //cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dev_texture, &num_bytes, cuda_texture);
    

    


    cudaStatus = cudaGraphicsUnmapResources(1, &cuda_texture, 0);

    cudaFree(dev_texture);
    dev_texture = 0;


}

void initCuda() {
    std::cout << "initCuda" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaGraphicsGLRegisterBuffer(&cudavbo0, VBO[0], cudaGraphicsMapFlagsWriteDiscard);
    //cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_texture, texture, cudaGraphicsMapFlagsWriteDiscard);
    cudaStatus = cudaGraphicsGLRegisterImage(&cuda_texture, texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);

    //moveleft();
    
    int a = 0;
    int b = 0;



}


void initWorld() {
    world = new sphere(vec3(0,0,-1), 0.5);
    

}

int main(int argc, char** argv) {
    initGL(&argc, argv);
    initCuda();
    initWorld();
    render();
	//glutInit(&argc, argv);
	//glutCreateWindow("OpenGL Window");
	//glutDisplayFunc(display);
	//glewInit();
	glutMainLoop();
	return 0;
}