#pragma once

#include "LinAlg.hpp"

class Relu
{
public:
        void forward(vecX<double> &input)
        {
            for(int i = 0; i < input.len; i++)
            {
                if(input.Get(i) < 0)
                    input.push(i, 0);
            }
        }
private:
};