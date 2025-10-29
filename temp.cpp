#include "Linear.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "MSE.hpp"
#include "CrossEntropyLoss.hpp"


using namespace std;

int main()
{
    Softmax sf;
    Linear li(5, 6);
    Relu rl;
    MSE mse;

    vecX<double> mat = RandomVecX(5, 4);

    mat.print();
    li.forward(mat);
    mat.print();
    rl.forward(mat);
    mat.print();
    sf.forward(mat);
    mat.print();
    vecX<double> actual = RandomVecX(6, 4);    
    vecX<double> vec = mse.forward(mat, actual);
    vec.print();

    return 0;
}