#include <iostream>
#include "LinAlg.hpp"

int main()
{
    vecX<double> mat2 = RandomVecX(4, 3);
    mat2.print();
    mat2.TR();
    mat2.print();
    mat2.TR();
    mat2.push(3, 2, 0);
    mat2.print();
    mat2.size().print();


    vecX<double> mat1(4, 3, 1);
    vecX<double> mat3(4, 3, 2);
    MatAdd(mat1, mat3).print();

    
    return 0;
}