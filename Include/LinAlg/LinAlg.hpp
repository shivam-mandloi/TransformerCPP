#pragma once

// #include <functional>
#include "vex.hpp"

vex<double> MatMul(vex<double> mat1, vex<double> mat2)
{
    vex<double> res(mat1.row, mat2.col, 0.0);
    for(int i = 0; i < mat1.row; i++)
    {
        for(int j = 0; j < mat2.col; j++)
        {
            double temp = 0;
            for(int k = 0; k < mat2.row; k++)
            {                
                temp += (mat1.Get(i, k) * mat2.Get(k, j));
            }
            res.push(i, j, temp);
        }
    }
    return res;
}