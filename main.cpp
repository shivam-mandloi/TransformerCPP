#include "Linear.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "MSE.hpp"
#include "CrossEntropyLoss.hpp"

using namespace std;
int main()
{
    Linear ly(4, 3);
    Relu rl;
    Softmax sf(1);
    MSE mse;
    CrossEntropyLoss cE;

    vecX<double> input = RandomVecX(3, 1);
    vecX<double> actual = RandomVecX(3, 1);

    input.print();

    cout << "After Linear Layer" << endl;
    ly.forward(input);
    input.print();

    cout << "After Relu Layer" << endl;
    rl.forward(input);
    input.print();

    cout << "After Sfotmax Layer" << endl;
    sf.forward(input);
    input.print();

    actual.print();
    input.print();

    sf.forward(input);
    sf.forward(actual);

    cout << "Loss: " << mse.forward(input, actual) << endl;
    cout << "Loss: " << cE.forward(input, actual) << endl;

    return 0;
}