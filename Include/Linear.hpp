#pragma once

#include "LinAlg.hpp"

/*
    forward -> Input <column vector> / return <column vector>

    backward -> Input <column vector> / return <column vector>
*/

class Linear
{
public:
    Linear(int inFeature, int outFeature)
    {
        weight = RandomVecX(outFeature, inFeature, 0.0, 0.1);
        bias = RandomVecX(outFeature, 1, 0.0, 0.1);
    }

    void forward(vecX<double> &input)
    {
        saved = input;
        input = MatAdd(MatMul(weight, input), bias);
    }

    void backward(vecX<double> &prevGrad)
    {
        saved.TR(); // saved vector is column vector

        // Find the gradient of loss w.r.t. weight matrix
        vecX<double> weigthUpdate = MatMul(prevGrad, saved);

        // Find the gradient of loss w.r.t. bias
        vecX<double> bias = prevGrad;

        // Find the gradient of loss w.r.t. input
        prevGrad = MatMul(prevGrad, weight);

        prevGrad.TR(); // prevGrad by above operation is row vector
    }

private:
    vecX<double> weight, bias, saved;
};