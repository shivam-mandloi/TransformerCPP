#include "Transformer.hpp"
#include "LinAlg.hpp"


using namespace std;

int main()
{
    Attention attn(3, 4, 5, 2);
    vecX<double> a(3, 10, 1);
    vecX<double> atVec = attn.forward(a);
    atVec.print();
    return 0;
}