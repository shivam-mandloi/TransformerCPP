#pragma once

#include "LinAlg.hpp"

class MSE
{
public:
    vecX<double> forward(vecX<double> &input, vecX<double> &actual)
    {
         /*
            => Assume input in column vector
            (Input) m X n =>
                m = dim of input
                n = batch size
            => return sinlge vector
        */
        savedActual = actual; savedInput = input;
        vecX<double> loss(1, actual.col, 0);
        for(int i = 0; i < input.col; i++)
        {
            double sumLoss = 0.0;
            for(int j = 0; j < input.row; j++)
            {
                sumLoss += (actual.Get(j, i) - input.Get(j, i)) * (actual.Get(j, i) - input.Get(j, i));
            }
            loss.push(i, sumLoss / input.row);
        }
        
        return loss;
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