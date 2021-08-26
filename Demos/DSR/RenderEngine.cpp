#include "RenderEngine.h"
#include <iostream>
#include "gl/glew.h"
#include <GL/freeglut.h>

#include "gl/freeglut.h"
static int window_width = 512;
static int window_height = 512;
static RenderEngine* engine = nullptr;

void display()
{
    engine->display1();
}
void RenderEngine::display1()
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


bool RenderEngine::initGL(int* argc, char** argv)
{
    std::cout << "initGL" << std::endl;
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    ////glutKeyboardFunc(keyboard);
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

    // viewport
    glViewport(0, 0, window_width, window_height);

    //// projection
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

    mesh = Mesh::CreateQuadFlipY(glm::vec4(0, 0, 0.5, 0.5));
    texture = new Texture(window_width, window_height, Texture::Usage::HeightMap);
    shader = new Shader("abc");
    shader->Load2("PureColor3D_vert.shader", "PureColor3D_frag.shader");
    material = new Material(shader);
    material->configStatus = Material::ConfigStatus_Geomtery;
    material->SetTexture("_HeightMap", texture);


    

    //// SDK_CHECK_ERROR_GL();
    //std::cout << "initGL succeed! " << std::endl;
    return true;
}
RenderEngine::RenderEngine(int* argc, char** argv)
{
	std::cout << "RenderEngine::RenderEngine" << std::endl;
    engine = this;
    initGL(argc, argv);
}

void RenderEngine::run()
{
    glutMainLoop();
}
