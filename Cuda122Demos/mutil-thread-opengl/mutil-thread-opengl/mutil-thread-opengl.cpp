
// https://www.glfw.org/docs/latest/context_guide.html#context_offscreen
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <cmath>

GLFWwindow* window = nullptr;
GLFWwindow* second_window = nullptr;
GLuint vao;
GLuint vbo;
bool running = true;

void thread_func() {
	glfwMakeContextCurrent(second_window);
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW in second thread" << std::endl;
		return;
	}

	GLfloat vertices[] = {
	-0.5f,-0.5f,0.0f, // top
	0.5f,-0.5f,0.0f, // right
	1.0f,0.5f,0.0f // left
	};

	// while (true) {
	// 	static int angle = 0;
	// 	angle++;
	// 	vertices[6] = cos(angle * 3.1415926 / 180);
	// 	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// 	// convert to float array
	// 	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	// }


	while (running) {
		static int angle = 0;
		angle++;
		vertices[6] = cos(angle * 3.1415926 / 180);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// convert to float array
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);


		// glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		// glClear(GL_COLOR_BUFFER_BIT);
		// glColor3f(0, 1, 0);
		// glEnableVertexAttribArray(0);

		// glVertexAttribPointer(
		// 	0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		// 	3,                  // size
		// 	GL_FLOAT,           // type
		// 	GL_FALSE,           // normalized?
		// 	0,                  // stride
		// 	(void*)0            // array buffer offset
		// );
		// glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// draw triangle
		// glDrawArrays(GL_TRIANGLES, 0, 3);

		// glBegin(GL_TRIANGLES);
		// glVertex3f(-0.5f,-0.5f,0.0f);
		// glVertex3f(0.5f,-0.5f,0.0f);
		// glVertex3f(0.0f,0.5f,0.0f);
		// glEnd();
		glfwSwapBuffers(second_window);
		// glfwPollEvents();
	}

}
int main() {
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return 1;
	}





	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(640, 480, "GLFW window", nullptr, nullptr);
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	second_window = glfwCreateWindow(640, 480, "Second Window", NULL, window);
	if (!window) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return 1;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		return 1;
	}

	// get version of opengl
	std::cout << glGetString(GL_VERSION) << std::endl;
	// std::cout << glGetInteger(GLEW_VERSION_MAJOR) << std::endl;

	// create vao

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);


	// create vbo

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// create a triangle
	GLfloat vertices[] = {
		-0.5f,-0.5f,0.0f, // top
		0.5f,-0.5f,0.0f, // right
		0.0f,0.5f,0.0f // left
	};

	// convert to float array
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	std::thread th(thread_func);



	while (!glfwWindowShouldClose(window)) {
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(0, 1, 0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(
			0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// draw triangle
		glDrawArrays(GL_TRIANGLES, 0, 3);

		// glBegin(GL_TRIANGLES);
		// glVertex3f(-0.5f,-0.5f,0.0f);
		// glVertex3f(0.5f,-0.5f,0.0f);
		// glVertex3f(0.0f,0.5f,0.0f);
		// glEnd();
		glfwSwapBuffers(window);
		glfwPollEvents();
	};
	running = false;
	th.join();
	glfwTerminate();

	int a = 0;
	int b = 1;

	std::cout << "hello world!!" << std::endl;
	return 0;
}
