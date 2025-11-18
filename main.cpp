#include "Linear.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "MSE.hpp"
#include "CrossEntropyLoss.hpp"

using namespace std;

int main()
{
    Linear ly(5, 10);
    vecX<double> input = RandomVecX(5, 1);
    ly.forward(input);
}