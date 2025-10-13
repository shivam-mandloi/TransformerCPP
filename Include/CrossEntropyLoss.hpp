#pragma once

#include "LinAlg.hpp"

class CrossEntropyLoss
{
public:
    CrossEntropyLoss(){}

    double forward(vecX<double> &input, vecX<double> &actual)
    {
        double loss = 0;
        for(int i = 0; i < input.len; i++)
        {
            loss += (actual.Get(i) * std::log(input.Get(i)));
        }
        return -loss;
    }

private:

};