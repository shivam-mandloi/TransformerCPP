#pragma once

#include "SGD.hpp"
#include "RMSProp.hpp"

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
RMSprop rMSprop;
public:
    Optim(OptimType _type, double _lr) : type(_type), lr(_lr), rMSprop(lr)
    {}

    void update(vecX<double> &weight, vecX<double> &weightGrad, vecX<double> &bias, vecX<double> &biasGrad)
    {
        if(type == SGD_O)
        {
            SGD sgd(lr);
            sgd.update(weight, weightGrad, bias, biasGrad);
        }
        if(type == RMSPROP_O)
        {
            rMSprop.update(weight, weightGrad, bias, biasGrad);
        }
    }
};