/*
    Include code to find the backpropagation    
*/


#pragma once

#include "vecX.hpp"
#include "LinAlg.hpp"
#include <random>
#include <ctime>
#include <cmath>
#include <functional>

void LinearBackward(vecX<double> &prevGrad, vecX<double> &weight, vecX<double> &bias, vecX<double> &saved, vecX<double> &weigthUpdate, vecX<double> &biasUpdate)
{
    /*
        y = Wx + b
            x = saved (n X m)
            b = bias (n)
            W = weight (k X n)

            n = Input Dim
            m = Batch Size
            k = Output Dim

        (Input):  prevGrad (m X k), weight (k X n), bias (n), saved (n X m), weigthUpdate (k X n), biasUpdate (n)

        output: update the, prevGrad (m X k) -> (m X n), weigthUpdate (k X n), biasUpdate (n)
    */

    // Find the gradient of loss w.r.t. weight matrix
    // we find for each element in batch and then take the mean

    prevGrad.TR(); // assume that input prevgrad is row vector
    saved.TR();

    // for each input in batch we find the grad and then use it's mean to update the weights
    for (int i = 0; i < prevGrad.col; i++)
    {
        for (int j = 0; j < prevGrad.row; j++)
        {
            for (int k = 0; k < saved.col; k++)
            {
                weigthUpdate.push(j, k, saved.Get(j, k) + (prevGrad.Get(j, i) * saved.Get(i, k)));
            }
        }
    }

    // Bias
    for (int i = 0; i < prevGrad.col; i++)
    {
        for (int j = 0; j < prevGrad.row; j++)
        {
            biasUpdate.push(j, 1, biasUpdate.Get(j, 1) + prevGrad.Get(j, i));
        }
    }

    // Divide by the bath size
    MatScalarProd(weigthUpdate, 1 / prevGrad.col);
    MatScalarProd(biasUpdate, 1 / prevGrad.col);

    prevGrad.TR();
    prevGrad = MatMul(prevGrad, weight);
}

void SoftMaxGrad(vecX<double> &prevGrad, vecX<double> &prob)
{
    /*
        (prevGrad): m X n
        (prob): n X m

            n = Input Dim
            m = Batch Size

        (Return): (prevGrad) m X n
    */
    
    // For each batch we find the derivative
    for(int batch = 0; batch < prevGrad.row; batch++) // m => batch size times
    {
        // for single batch input
        for (int i = 0; i < prevGrad.col; i++) // n => input dim times
        {
            double total = 0.0;
            for (int j = 0; j < prob.row; j++) // n => input dim times
            {
                // -p_i * p_j
                double p = -prob.Get(i, batch) * prob.Get(j, batch);
                if(i == j)
                    p = prob.Get(batch, i) + p; // p_i - p_i * p_j
                total += prevGrad.Get(batch, j) * p; // matrix multiplication (prevGrad @ grad_wrt_softmax)
            }
            prevGrad.push(batch, i, total);
        }
    }
}