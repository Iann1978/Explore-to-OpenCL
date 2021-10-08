
//#include "SysInclude\\GL\\glew.h"
#include "Shader.h"
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <istream>
#include <sstream>
#include <vector>

static std::string LoadShaderSources(int count, ...)
{
	std::stringstream ss;
	va_list args;
	va_start(args, count);
	while (count--)
	{
		const char* filename = va_arg(args, const char*);
		printf("Load shader file: %s\n", filename);
		std::ifstream is(filename, std::ios::in);
		ss << is.rdbuf();
		is.close();
	}
	va_end(args);
	return ss.str();
}

int GenGLShader(int type, const char* source)
{
	GLint Result = GL_FALSE;
	int InfoLogLength = 0;

	GLuint shaderId = glCreateShader(type);
	shaderId = glCreateShader(type);

	// Compile Vertex Shader
	printf("Compiling shader\n");
	glShaderSource(shaderId, 1, &source, NULL);
	glCompileShader(shaderId);

	// Check Vertex Shader
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ErrorMessage(InfoLogLength + 1);
		glGetShaderInfoLog(shaderId, InfoLogLength, NULL, &ErrorMessage[0]);
		printf("%s\n", &ErrorMessage[0]);
	}
	return shaderId;
}

int GenGLProgram(const char* vs, const char* fs,
	const char* gs = nullptr,
	const char* tcs = nullptr,
	const char* tes = nullptr)
{
	GLint Result = GL_FALSE;
	int InfoLogLength = 0;

	GLuint vsId = 0, fsId = 0, gsId = 0, tcsId = 0, tesId = 0;
	
	vsId = GenGLShader(GL_VERTEX_SHADER, vs);
	fsId = GenGLShader(GL_FRAGMENT_SHADER, fs);
	if (gs && strlen(gs)) gsId = GenGLShader(GL_GEOMETRY_SHADER, gs);
	if (tcs && strlen(tcs)) tcsId = GenGLShader(GL_TESS_CONTROL_SHADER, tcs);
	if (tes && strlen(tes)) tesId = GenGLShader(GL_TESS_EVALUATION_SHADER, tes);


	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, vsId);
	glAttachShader(ProgramID, fsId);
	if (gsId) glAttachShader(ProgramID, gsId);
	if (tcsId) glAttachShader(ProgramID, tcsId);
	if (tesId) glAttachShader(ProgramID, tesId);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}


	glDetachShader(ProgramID, vsId);
	glDetachShader(ProgramID, fsId);
	if (gsId) glDetachShader(ProgramID, gsId);
	if (tcsId) glDetachShader(ProgramID, tcsId);
	if (tesId) glDetachShader(ProgramID, tesId);

	glDeleteShader(vsId);
	glDeleteShader(fsId);
	if (gsId) glDeleteShader(gsId);
	if (tcsId) glDeleteShader(tcsId);
	if (tesId) glDeleteShader(tesId);

	return ProgramID;

}
Shader::Shader(const char* name)
{
	strcpy_s(this->name, sizeof(this->name) - 1, name);

	//shaders.insert(std::pair<std::string, Shader*>(this->name, this));
}
Shader::Shader(const char* name,
	const char* vertexShaderFile,
	const char* fragmentShaderFile,
	const char* geometryShaderFile,
	const char* vertexIncludeFile,
	const char* fragmentIncludeFile,
	const char* geometryIncludeFile
)
{
	strcpy_s(this->name, sizeof(this->name)-1, name);
	std::string vertexShaderSource, fragmentShaderSource, geometryShaderSource;
	vertexShaderSource = vertexIncludeFile ? LoadShaderSources(2, vertexIncludeFile, vertexShaderFile) : LoadShaderSources(1, vertexShaderFile);
	fragmentShaderSource = fragmentIncludeFile ? LoadShaderSources(2, fragmentIncludeFile, fragmentShaderFile) : LoadShaderSources(1, fragmentShaderFile);
	if (geometryShaderFile)
	{
		geometryShaderSource = geometryIncludeFile? LoadShaderSources(2, geometryIncludeFile, geometryShaderFile): LoadShaderSources(1, geometryShaderFile);
	}
	//program = LoadShaderFromSourceCode(vertexShaderSource.c_str(), fragmentShaderSource.c_str(),
	//	geometryShaderFile ? geometryShaderSource.c_str() : nullptr);
	program = GenGLProgram(vertexShaderSource.c_str(), fragmentShaderSource.c_str(),
		geometryShaderSource.c_str());
	//program = LoadShaders(vertexShaderFile, fragmentShaderFile, geometryShaderFile);
	//shaders.insert(std::pair<std::string, Shader*>(this->name, this));
}


Shader::~Shader()
{
	glDeleteProgram(program);
}

void Shader::Load(const char* incfile, const char* vsfile, const char* fsfile,
	const char* gsfile,
	const char* tcsfile,
	const char* tesfile)
{
	std::string vsSource, fsSource, gsSource, tcsSource, tesSource;
	if (incfile)
	{
		vsSource = LoadShaderSources(2, incfile, vsfile);
		fsSource = LoadShaderSources(2, incfile, fsfile);
		if (gsfile) gsSource = LoadShaderSources(2, incfile, gsfile);
		if (tcsfile) tcsSource = LoadShaderSources(2, incfile, tcsfile);
		if (tesfile) tesSource = LoadShaderSources(2, incfile, tesfile);
	}
	else
	{
		vsSource = LoadShaderSources(1, vsfile);
		fsSource = LoadShaderSources(1, fsfile);
		if (gsfile) gsSource = LoadShaderSources(1, gsfile);
		if (tcsfile) tcsSource = LoadShaderSources(1, tcsfile);
		if (tesfile) tesSource = LoadShaderSources(1, tesfile);
	}

	program = GenGLProgram(vsSource.c_str(), fsSource.c_str(),
		gsSource.c_str(), tcsSource.c_str(), tesSource.c_str());

}

static void LoadShader(const char* filename, std::stringstream& ss)
{
	std::fstream fin(filename, std::ios_base::in); 
	std::string ReadLine;
	while (getline(fin, ReadLine))
	{
		if (0 != strcmp(ReadLine.substr(0,8).c_str(), "#include"))
		{
			ss << ReadLine << std::endl;
			continue;
		}
			
		std::string incFilename; {
			int pos0 = ReadLine.find_first_of('"')+1;
			int pos1 = ReadLine.find_last_of('"');
			incFilename = ReadLine.substr(pos0, pos1 - pos0);
		}
		LoadShader(incFilename.c_str(), ss);
	}
}

void Shader::Load2(const char* vsfile, const char* fsfile,
	const char* gsfile,	const char* tcsfile, const char* tesfile)
{
	std::stringstream ssVs, ssFs, ssTcs, ssTes;

	LoadShader(vsfile, ssVs);
	LoadShader(fsfile, ssFs);
	if (tcsfile) LoadShader(tcsfile, ssTcs);
	if (tesfile) LoadShader(tesfile, ssTes);
	program = GenGLProgram(ssVs.str().c_str(), ssFs.str().c_str(), 0,
		ssTcs.str().c_str(), ssTes.str().c_str());
}
GLuint Shader::GetLocation(const char *valuename)
{
	return glGetUniformLocation(program, valuename);
}


