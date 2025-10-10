#include <iostream>
#include "LinAlg.hpp"

int main() {
    vex<double> mat2(1, 3, 2);
    vex<double> mat1(4, 3, 3);
    
    mat2.TR(); // Transpose

    mat1.print();
    mat2.print();

    mat1.size().print();
    mat2.size().print();

    MatMul(mat1, mat2).print();
    return 0;
}