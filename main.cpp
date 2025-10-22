#include "Linear.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "MSE.hpp"
#include "CrossEntropyLoss.hpp"

using namespace std;

// One Hidden Layer network
struct NeuralNetwork
{
    // we can change the optimization to ADAM_O/RMSPROP_O
    NeuralNetwork(int inFeature, int hiddenFeature, int outFeature) : inputDim(inFeature), hiddenDim(hiddenFeature), outDim(outFeature), ly1(inFeature, hiddenDim), ly2(hiddenDim, outDim)
    {}

    double Train(vecX<double> input, int index)
    {
        Predict(input);

        // predict the output
        double loss = crLoss.forward(input, index);

        // Back propagation
        vecX<double> prevGrad = crLoss.backward();
        sf.backward(prevGrad);
        ly2.backward(prevGrad);
        rl.backward(prevGrad);
        ly1.backward(prevGrad);

        // update the parameters
        ly1.update();
        ly2.update();

        return loss;
    }

    void Predict(vecX<double> &input)
    {
        ly1.forward(input);
        rl.forward(input);
        ly2.forward(input);
        sf.forward(input);
        cout << endl;
        input.print();
    }

private:
    int inputDim, hiddenDim, outDim;
    Linear ly1, ly2;
    Relu rl;
    Softmax sf;
    CrossEntropyLoss crLoss;
};

int main()
{
    // Create one hidden layer model
    NeuralNetwork nn(5, 10, 3);

    // Initialize the data
    vecX<double> input = {10, 4, 1, 3, 2};
    vecX<double> trueRes = {0, 0, 1};


    // Try to overfit model with the same input and output
    for(int i = 0; i < 100; i++)
    {
        vecX<double> temp = input;
        temp.TR();
        cout << nn.Train(temp, 2) << endl;
    }
}