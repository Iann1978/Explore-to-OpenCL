#version 330 core

uniform mat4 mvp = mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(0, 0, 0, 1));

uniform mat4 mv = mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(0, 0, 0, 1));


uniform mat4 p = mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(0, 0, 0, 1));

layout(location = 0) in vec3 input_VertexPosition;



void main() {
	gl_Position = p * mv * vec4(input_VertexPosition,1);
}

