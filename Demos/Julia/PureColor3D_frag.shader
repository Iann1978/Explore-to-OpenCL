#version 330 core

uniform vec4 color = vec4(1,1,0,1);
uniform sampler2D _HeightMap;

out vec4 outColor;
in vec2 texcoord0;

void main(){
	outColor = texture(_HeightMap, texcoord0);
	//outColor = color;
}