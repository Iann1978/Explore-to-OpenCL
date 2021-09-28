//²Î¿¼ÎÄµµ£ºE:\work\githome\opencl\cuda-sample\2_Graphics\simpleGL/simpleGL.cu
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include "gl/glew.h"
#include "gl/freeglut.h"
static int window_width = 800;
static int window_height = 600;
#include "Shader.h"
#include "Material.h"
#include "Mesh.h"
Shader* shader = nullptr;
Material* material = nullptr;
Mesh* mesh = nullptr;
int main1();
void display()
{
  //  main1();
    //sdkStartTimer(&timer);

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

    //g_fAnim += 0.01f;

    //sdkStopTimer(&timer);
    //computeFPS();
}


bool initGL(int* argc, char** argv)
{
    std::cout << "initGL" << std::endl;
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    //glutKeyboardFunc(keyboard);
    //glutMotionFunc(motion);
    //glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
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

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);



    shader = new Shader("abc");
    shader->Load2("PureColor3D_vert.shader", "PureColor3D_frag.shader");
    material = new Material(shader);
    material->configStatus = Material::ConfigStatus_Geomtery;
    mesh = Mesh::CreateQuadFlipY(glm::vec4(0, 0, 0.5, 0.5));


    // SDK_CHECK_ERROR_GL();
    std::cout << "initGL succeed! " << std::endl;
    return true;
}


int main1();
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
cudaGraphicsResource* vbo_res = nullptr;
bool launch_kernel(float* pos);
bool initCuda()
{
    std::cout << "initCuda" << std::endl;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

   ;

    cudaStatus = cudaGraphicsGLRegisterBuffer(&vbo_res, mesh->vertexbuffer, cudaGraphicsMapFlagsWriteDiscard);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsGLRegisterBuffer failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    cudaStatus = cudaGraphicsMapResources(1, &vbo_res, 0);
  //  checkCudaErrors(cudaStatus);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsMapResources failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    float* dptr = 0;
    size_t num_bytes = 0;


    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, vbo_res);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsResourceGetMappedPointer failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    if (!launch_kernel(dptr))
    {
        goto Error;
    }

    cudaStatus = cudaGraphicsUnmapResources(1, &vbo_res, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGraphicsUnmapResources failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
      //  launch_kernel
    std::cout << "initCuda succeed!" << std::endl;
    return true;
Error:
    return false;
    
}

#include "GLCudaEngine.h"

int main(int argc, char** argv)
{
    GLCudaEngine engine;

    engine.InitGL(&argc, argv);
    engine.InitCuda();
    engine.Run();


	std::cout << "GlCuda1" << std::endl;



    if (!initGL(&argc, argv))
        return 1;


    if (!initCuda())
        return 2;
     
    glutMainLoop();
    //main1();
  
    // 

	return 0;
}