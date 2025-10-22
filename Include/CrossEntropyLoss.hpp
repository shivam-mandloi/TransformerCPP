#pragma once

#include "LinAlg.hpp"

class CrossEntropyLoss
{
public:
    CrossEntropyLoss(){}

    double forward(vecX<double> &pred, int actual)
    {
        // pred should be column vector
        savedPred = pred; index = actual; // save probability and actual index
        double loss = -std::log(pred.Get(actual));
        return loss;
    }

    vecX<double> backward()
    {
        // return column vector
        vecX<double> grad(savedPred.len, 1, 0);
        grad.push(index, -1/savedPred.Get(index));
        return grad;        
    }

private:
    vecX<double> savedPred;
    int index;
};