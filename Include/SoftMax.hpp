#pragma once

#include "LinAlg.hpp"
#include <functional>

/*
    forward -> Input <column vector> / return <column vector>

    backward -> Input <column vector> / return <column vector>
*/

std::function<double(double, double, double)> clip = [](double minVal, double maxVal, double val){
    return val > maxVal ? maxVal : (minVal > val ? minVal : val);
};

class Softmax
{
public:
    Softmax(double _temprature = 1): temprature(_temprature){}
    void forward(vecX<double> &input)
    {
        double total = 0.0;
        
        for(int i = 0; i < input.len; i++)
        {
            double temp = std::exp(clip(-50, 50, input.Get(i)/temprature));
            input.push(i, temp);
            total += temp;
        }

        for(int i = 0; i < input.len; i++)
            input.push(i, input.Get(i)/total);
        prob = input;
    }

    void backward(vecX<double> &prevGrad)
    {
        // we assume that prevGrad is column vector
        vecX<double> derMat(prevGrad.len, prevGrad.len, 0);
        
        // grad dy/dx
        for(int i = 0; i < prevGrad.len; i++)
        {
            for(int j = 0; j < prevGrad.len; j++)
            {
                if(i == j)
                {
                    derMat.push(i, j, prob.Get(i) * (1- prob.Get(i)));
                    continue;
                }
                derMat.push(i, j, - prob.Get(i) * prob.Get(j));
            }
        }
        prevGrad.TR();
        prevGrad = MatMul(prevGrad, derMat);
    }

private:
    double temprature;
    vecX<double> prob;
};