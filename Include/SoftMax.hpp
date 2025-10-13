#pragma once

#include "LinAlg.hpp"
#include <functional>

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
    }

private:
    double temprature;
};