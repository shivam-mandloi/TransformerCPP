#pragma once

#include "LinAlg.hpp"
#include "Optim.hpp"

/*
    forward -> Input <column vector> / return <column vector>

    backward -> Input <column vector> / return <column vector>
*/

class Linear
{
public:
    Optim opt;
    Linear(int inFeature, int outFeature, double lr = 0.01)
    {
        weight = RandomVecX(outFeature, inFeature, 0.0, 0.1);
        bias = RandomVecX(outFeature, 1, 0.0, 0.1);        
    }

    void forward(vecX<double> &input)
    {
        saved = input;
        input = MatMul(weight, input); // wx
        input = MatAdd(input, bias); // wx + b
    }

    void backward(vecX<double> &prevGrad)
    {
        saved.TR(); // saved vector is column vector

        // Find the gradient of loss w.r.t. weight matrix
        weigthUpdate = MatMul(prevGrad, saved); // saved weight grad

        // Find the gradient of loss w.r.t. bias
        biasUpdate = prevGrad; // saved bias grad

        prevGrad.TR(); // make it row vector to multiply with weight

        // Find the gradient of loss w.r.t. input
        prevGrad = MatMul(prevGrad, weight);

        prevGrad.TR(); // prevGrad by above operation is row vector
    }

    void update()
    {
        // update the weights
        opt.update(weight, weigthUpdate, bias, biasUpdate);
        // weight.size().print();
        // weigthUpdate.size().print();
        // bias.size().print();
        // biasUpdate.size().print();
    }

private:
    vecX<double> weight, bias, saved, weigthUpdate, biasUpdate;
};