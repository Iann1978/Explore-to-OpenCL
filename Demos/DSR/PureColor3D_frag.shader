#version 330 core

uniform vec4 color = vec4(1,1,0,1);
uniform sampler2D _HeightMap;

out vec4 outColor;

void main(){
	outColor = texture(_HeightMap, vec2(0.5,0.5));
	outColor = color;
}