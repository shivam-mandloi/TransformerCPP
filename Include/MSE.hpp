#pragma once

#include "LinAlg.hpp"

class MSE
{
public:
    double forward(vecX<double> &input, vecX<double> &actual)
    {
        // both input and actual should be column vector
        // input here define the predicted vector
        savedActual = actual; savedInput = input;
        double loss = 0;
        
        for(int i = 0; i < input.len; i++)
        {
            loss += (actual.Get(i) - input.Get(i)) * (actual.Get(i) - input.Get(i));
        }
        
        return loss / input.len;
    }

    vecX<double> backward()
    {
        // column vector
        vecX<double> res(savedActual.len, 1, 0);
        double len = 2/(double)savedActual.len;
        
        for(int i = 0; i < savedActual.len; i++)
        {
            res.push(i, len * (savedInput.Get(i) - savedActual.Get(i)));
        }

        return res;
    }

private:
    vecX<double> savedInput, savedActual;
};