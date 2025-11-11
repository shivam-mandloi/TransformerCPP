#include "Linear.hpp"
#include "Relu.hpp"
#include "SoftMax.hpp"
#include "MSE.hpp"
#include "CrossEntropyLoss.hpp"
#include "ReadData.hpp"

using namespace std;

struct NeuralNetwork
{
    NeuralNetwork(int inFeature, int hiddenFeature, int outFeature) : ly1(inFeature, hiddenFeature), ly2(hiddenFeature, outFeature)
    {
        // Set the Optimizer
        ly1.opt.SetAdam(0.99, 0.99, 1e-3);
        ly2.opt.SetAdam(0.99, 0.99, 1e-3);
        // ly1.opt.SetRMSprop(0.99, 0.001);
        // ly2.opt.SetRMSprop(0.99, 0.001);
        // ly1.opt.SetSGD(0.01);
        // ly2.opt.SetSGD(0.01);
    }

    // Training Code
    void Train(vector<pair<vecX<double>, vecX<int>>> &batchData, int epoch)
    {
        for(int epc = 0; epc < epoch; epc++)
        {
            for(int batch = 0; batch < batchData.size(); batch++)
            {
                vecX<double> input = batchData[batch].first;
                vecX<int> target = batchData[batch].second;
                input.TR();
    
                vecX<double> loss = Predict(input, target); // Return Loss

                cout << "Loss: ";
                loss.print();
                cout << " Batch: " << batch + 1 << " Epoch: " << epc + 1 << endl;

                // Backpropagation
                vecX<double> prevGrad = crLoss.backward();
                
                sf.backward(prevGrad);
                ly2.backward(prevGrad);
                rl.backward(prevGrad);
                ly1.backward(prevGrad);

                ly1.update(); ly2.update();
            }
        }
    }

    vecX<double> Predict(vecX<double> &input, vecX<int> &target)
    {
        ly1.forward(input);
        rl.forward(input);
        ly2.forward(input);
        sf.forward(input);
        return crLoss.forward(input,target);
    }

private:
    Linear ly1, ly2;
    Relu rl;
    Softmax sf;
    CrossEntropyLoss crLoss;
};

int main()
{
    // first line consist vector with size 1
    vector<vector<double>> data = ReadFile("C:\\Users\\shiva\\Desktop\\IISC\\code\\NN Model C++\\Basic NN Blocks\\Iris\\Dataset\\Iris.csv");
    // Split data into batches
    vector<pair<vecX<double>, vecX<int>>> batchedData = CreateBatch(data, 16);

    
    NeuralNetwork nn(4, 10, 3);
    // Start Training
    nn.Train(batchedData, 1000);

    // // print all data
    // for(int i = 0; i < data.size(); i++)
    // {
    //     for(int j = 0; j < data[i].size(); j++)
    //     {
    //         cout << data[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // // Print Batched Data
    // for(int i = 0; i < batchedData.size(); i++)
    // {
    //     cout << "Input Feature" << endl;
    //     batchedData[i].first.print();
    //     cout << "Target Feature" << endl;
    //     batchedData[i].second.print();
    // }
}