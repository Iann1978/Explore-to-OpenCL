#include "GLCudaEngine.h"
#include <iostream>
#include "gl/glew.h"
#include "gl/freeglut.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

static int window_width = 800;
static int window_height = 600;


static GLCudaEngine* engine = nullptr;
void display2()
{
    engine->display1();
}
GLCudaEngine::GLCudaEngine()
{
    engine = this;

}

void GLCudaEngine::InitGL(int* argc, char** argv)
{
    std::cout << "initGL" << std::endl;
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display2);
    glutIdleFunc(display2);
    //glutKeyboardFunc(keyboard);
    //glutMotionFunc(motion);
    //glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
       // return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);



    shader = new Shader("abc");
    shader->Load2("PureColor3D_vert.shader", "PureColor3D_frag.shader");

    texture = new Texture(512, 512, Texture::Usage::HeightMap);

    material = new Material(shader);
    material->configStatus = Material::ConfigStatus_Geomtery;
    material->SetTexture("_HeightMap", texture);
    mesh = Mesh::CreateQuadFlipY(glm::vec4(-1, -1, 2, 2));


    exchanger = new BufferExchanger(texture);
    // SDK_CHECK_ERROR_GL();
    std::cout << "initGL succeed! " << std::endl;
   // return true;
}
//struct cudaGraphicsResource* cuda_pbo_resource;
//GLuint pbo = 0;
float* dptr = 0;
size_t num_bytes = 0;
struct cudaGraphicsResource* cuda_pbo_resource = nullptr;


bool GLCudaEngine::InitCuda()
{
    std::cout << "DSRSimulator::DSRSimulator()" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, exchanger->pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!");
        goto Error;
    }
    return true;
Error:
    std::cout << "Error in DSRSimulator::DSRSimulator()" << std::endl;
    return false;
}
bool launch_kernel(float* pos, int width, int height);
void GLCudaEngine::UpdateCuda()
{
    std::cout << "DSRSimulator::runOnce()" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsMapResources failed!");
        goto Error;
    }


    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_pbo_resource);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!");
        goto Error;
    }

   bool err = launch_kernel(dptr, exchanger->tex->width, exchanger->tex->height);


    cudaStatus = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsUnmapResources failed!");
        goto Error;
    }

    int a = 0;
    int b = 0;
    return;
Error:
    std::cout << "Error in DSRSimulator::runOnce()" << std::endl;
}

void GLCudaEngine::display1()
{
    std::cout << __FUNCTION__ << std::endl;

    exchanger->ReadFromTexture(this->texture);

    UpdateCuda();
    //simulator->runOnce();

    exchanger->WriteToTexture(this->texture);


    //// run CUDA kernel to generate vertex positions
    //runCuda(&cuda_vbo_resource);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //// set view matrix
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();
    //glTranslatef(0.0, 0.0, translate_z);
    //glRotatef(rotate_x, 1.0, 0.0, 0.0);
    //glRotatef(rotate_y, 0.0, 1.0, 0.0);

    //// render from the vbo
    //glBindBuffer(GL_ARRAY_BUFFER, vbo);
    //glVertexPointer(4, GL_FLOAT, 0, 0);

    //glEnableClientState(GL_VERTEX_ARRAY);
    //glColor3f(1.0, 0.0, 0.0);
    //glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    //glDisableClientState(GL_VERTEX_ARRAY);

    material->Use();
    Mesh::RenderMesh(mesh);

    glutSwapBuffers();

}

void GLCudaEngine::Run()
{
    glutMainLoop();
}