// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#pragma pack(push, 1)
class A {
    
public:
    int a;
    virtual void func() { 
        std::cout << "calling func in a."; 
    };
};

class B {
    
public:
    int b;
    virtual void func() { 
        std::cout << "calling func in b.";
    };

};

class C : public A, public B {
public:
    int c;
    virtual void func() {
        std::cout << "calling func in c.";
	};

};
#pragma pack(pop, 1)
int main()
{
    C c;
    A* pa = &c;
    B* pb = &c;


    std::cout << "Address of pa is " << pa << std::endl;
    std::cout << "Address of pb is " << pb << std::endl;
    std::cout << "Address of c is " << &c << std::endl;

    std::cout << "Address of variable a is " << &(c.a) << std::endl;
    std::cout << "Address of variable b is " << &(c.b) << std::endl;
    std::cout << "Address of variable c is " << &(c.c) << std::endl;

    std::cout << "Sizeof A is " << sizeof(A) << std::endl;
std::cout << "Sizeof B is " << sizeof(B) << std::endl;
std::cout << "Sizeof C is " << sizeof(C) << std::endl;

std::cout << "Sizeof instance of A is " << sizeof(*pa) << std::endl;
std::cout << "Sizeof instance of B is " << sizeof(*pb) << std::endl;
std::cout << "Sizeof instance of C is " << sizeof(c) << std::endl;

    pa->func();
    pb->func();
    //std::cout << "Hello World!\n";
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
