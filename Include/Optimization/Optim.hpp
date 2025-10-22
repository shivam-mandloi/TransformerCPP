#pragma once

#include "SGD.hpp"

enum OptimType
{
    SGD_O,
    ADAM_O,
    RMSPROP_O
};

class Optim
{
OptimType type;
double lr;
public:
    Optim(OptimType _type, double _lr) : type(_type), lr(_lr)
    {}

    void update(vecX<double> &weight, vecX<double> &weightGrad, vecX<double> &bias, vecX<double> &biasGrad)
    {
        if(type == SGD_O)
        {
            SGD sgd(lr);
            sgd.update(weight, weightGrad, bias, biasGrad);
        }
    }
};