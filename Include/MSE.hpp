#pragma once

#include "LinAlg.hpp"

class MSE
{
public:
    double forward(vecX<double> &input, vecX<double> &actual)
    {
        double loss = 0;
        for(int i = 0; i < input.len; i++)
        {
            loss += (actual.Get(i) - input.Get(i)) * (actual.Get(i) - input.Get(i));
        }
        return loss / input.len;
    }
private:

};