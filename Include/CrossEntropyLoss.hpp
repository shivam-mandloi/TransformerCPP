#pragma once

#include "LinAlg.hpp"

class CrossEntropyLoss
{
public:
    CrossEntropyLoss(){}

    vecX<double> forward(vecX<double> &pred, vecX<int> &actual)
    {
        /*
            => Assume input in column vector
            (Input) m X n =>
                m = dim of input
                n = batch size
            => return column vector
        */
        vecX<double> loss(pred.col, 1, 0);
        savedPred = pred;
        index = actual;
        for(int i = 0; i < pred.col; i++)
        {
            loss.push(i, -std::log(pred.Get(actual.Get(i), i)));
        }
        return loss;
    }

    vecX<double> backward()
    {
        /*
           => savedPred dimension is like this
            (Input) m X n =>
                m = dim of input
                n = batch size
            
            => actual have this dim
            (Input) n => column vector
            
            => return: matrix (n X m)
        */
        vecX<double> grad(savedPred.col, savedPred.row, 0);
        for(int i = 0; i < savedPred.row; i++)
        {
            grad.push(i, index.Get(i), 1 / savedPred.Get(index.Get(i)));
        }
        return grad;
    }

private:
    vecX<double> savedPred;
    vecX<int> index;    
};