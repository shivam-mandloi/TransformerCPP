#include "Linear.hpp"

using namespace std;
int main()
{
    Linear ly(4, 1);

    vecX<double> input = RandomVecX(4, 1);
    input.print();

    ly.Forward(input);
    input.print();

    return 0;
}