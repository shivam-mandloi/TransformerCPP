#pragma once

#include "vecX.hpp"
#include <random>
#include <ctime>

int randCount = 0;

vecX<double> MatMul(vecX<double> mat1, vecX<double> mat2)
{
    vecX<double> res(mat1.row, mat2.col, 0.0);
    for (int i = 0; i < mat1.row; i++)
    {
        for (int j = 0; j < mat2.col; j++)
        {
            double temp = 0;
            for (int k = 0; k < mat2.row; k++)
            {
                temp += (mat1.Get(i, k) * mat2.Get(k, j));
            }
            res.push(i, j, temp);
        }
    }
    return res;
}

vecX<double> MatAdd(vecX<double> mat1, vecX<double> mat2)
{
    vecX<double> res(mat1.row, mat2.col, 0.0);
    for (int i = 0; i < mat1.row; i++)
    {
        for (int j = 0; j < mat1.col; j++)
        {
            res.push(i, j, mat1.Get(i, j) + mat2.Get(i, j));
        }
    }
    return res;
}

vecX<double> RandomVecX(int row, int col, double mean = 0.0, double variance = 1.0)
{
    vecX<double> vec(row, col, 0.0);
    std::random_device rd;
    std::time_t currentTime = std::time(nullptr);
    unsigned int uniqueNumber = static_cast<unsigned int>(currentTime) + randCount;
    randCount++;
    std::mt19937 gen(uniqueNumber);
    std::normal_distribution<> gaussian(mean, std::sqrt(variance));

    for (int i = 0; i < vec.len; i++)
    {
        vec.push(i, gaussian(gen));
    }
    return vec;
}