/*
To use this code follow this steps
    1. Put this file to same folder where script.py exists.
    2. Change the filename variable in script.py with this file name.
    3. Run the python file.
    4. Once python run successfully, one main.exe file will be generated.
*/

// Example code for some function of LinAlg/vecX

#include <iostream>
#include "LinAlg.hpp"

int main()
{
    vecX<double> mat2 = RandomVecX(4, 3);
    // mat2.print();
    mat2.TR();
    // mat2.print();
    mat2.TR();
    mat2.push(3, 2, 0);
    // mat2.print();
    // mat2.size().print();


    vecX<double> mat1(3, 4, 1);
    vecX<double> mat3(4, 3, 2);
    
    mat1.push(1, 3, -2);
    mat1.TR();
    // HadamardProduct(mat1, mat3).print();


    vecX<double> vec = {1};
    vec.TR();

    mat1.print();
    // mat2.print();
    vec.print();
    
    MatAdd(mat1, vec).print();
    return 0;
}