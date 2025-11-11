#pragma once
#include "SoftMax.hpp"
#include "Relu.hpp"
#include "Linear.hpp"



struct Attention
{
    Linear Q, K, V;
    Attention(int inputDim, int keyDim, int valueDim) : Q(inputDim, keyDim), K(inputDim, keyDim), V(inputDim, valueDim)
    {

    }
};

class Transformer
{
Attention atn;
public:

};