/*
To use this code follow this steps
    1. Put this file to same folder where script.py exists.
    2. Change the filename variable in script.py with this file name.
    3. Run the python file.
    4. Once python run successfully, one main.exe file will be generated.
*/

// Sample code of nn with RMSprop optimizer, used for testing on one sample


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
    {
        // Set optimization algorithm as a RMSprop
        ly1.opt.SetRMSprop(0.99, 0.01); // gamma, learning rate
        ly2.opt.SetRMSprop(0.99, 0.01); 
    }

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
        // input.print();
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
        cout << "Loss: " << nn.Train(temp, 2) << endl;
    }
}