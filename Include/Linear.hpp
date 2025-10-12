#pragma once

#include "LinAlg.hpp"

class Linear
{
public:
    Linear(int inFeature, int outFeature)
    {
        weight = RandomVecX(outFeature, inFeature, 0.0, 0.1);
        bias = RandomVecX(outFeature, 1, 0.0, 0.1);
    }

    void Forward(vecX<double> &input)
    {
        input = MatAdd(MatMul(weight, input), bias);
    }

private:
    vecX<double> weight, bias;
};