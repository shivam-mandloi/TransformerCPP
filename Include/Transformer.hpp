#pragma once
#include "SoftMax.hpp"
#include "Relu.hpp"
#include "Linear.hpp"
#include "AttentionMultiply.hpp"

/*
    -> Take column vector
    (Input): n X s
        n = Input Dim
        s = Sequence Size
    (Output): n X s
*/
struct Attention
{
    Linear Q, K, V, O;
    int nHead, kDim, vDim;
    Softmax sf;
    // key and query output dim are same
    // Define for number of head
    Attention(int inputDim, int keyDim, int valueDim, int noOfHead)
        : Q(inputDim, noOfHead * keyDim), K(inputDim, noOfHead * keyDim), V(inputDim, noOfHead * valueDim), O(noOfHead * valueDim, inputDim), nHead(noOfHead), kDim(keyDim), vDim(valueDim) {}

    vecX<double> forward(vecX<double> &input)
    {
        vecX<double> queryIn = input, keyIn = input, valueIn = input;
        Q.forward(queryIn);
        K.forward(keyIn);
        V.forward(valueIn);

        vecX<double> qK = AttentionQueryKeyMultiply(nHead, queryIn, keyIn); // Return: (h*s X s)
        qK.TR();
        qK = MatScalarProd(qK, 1/(std::pow(kDim, 0.5)));
        sf.forward(qK); // s X h*s

        qK.TR();
        valueIn.TR();

        valueIn = AttentionValueMultiply(nHead, qK, valueIn); // Return: (s X h*dv)
        valueIn.TR();

        O.forward(valueIn);
        return valueIn;
    }

    void backward()
    {

    }
};

struct MLP
{
    vecX<double> forward(vecX<double> &input)
    {

    }
    void backward()
    {

    }
};

class Transformer
{
    Attention atn;
public:
    Transformer(int inputDim, int keyDim, int valueDim, int noOfHead) : atn(inputDim, keyDim, valueDim, noOfHead)
    {}
    vecX<double> forward(vecX<double> &input)
    {

    }
    void backward()
    {

    }
};