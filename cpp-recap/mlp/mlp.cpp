#include <iostream>
#include "matrix.hpp"
#include "matrix.cpp"


class Linear {
private:
    size_t in_features;
    size_t out_features;
    matrix<float> weight;
    matrix<float> bias;

    Linear() {

    }

public:
    Linear(size_t in_features, size_t out_features) {
        cout << "Linear(int in_features, int out_features)" << endl;
        this->in_features = in_features;
        this->out_features = out_features;
        matrix<float> weight(in_features, out_features);
        matrix<float> bias(1, out_features);
        this->weight = weight;
        this->bias = bias;
    }

    Linear(const Linear& other) {

    }
    ~Linear() {

    }

    void rand_init() {
        this->weight.rand_init();
        this->bias.rand_init();
    }

    void init_normal() {

    }

    void init_uniform() {
        
    }

    matrix<float> forward(const matrix<float>& input) {
        matrix<float> output = input * this->weight + this->bias;
        return output;
    }

    void print() {
        cout << "weight: " << endl;
        this->weight.print();
        cout << "bias: " << endl;
        this->bias.print();
    }
};

int main() {
    matrix<float> input(1, 3);
    input.rand_init();
    cout << "input: " << endl;
    input.print();

    Linear linear(3, 2);
    linear.rand_init();
    cout << "Linear: " << endl;
    linear.print();

    matrix<float> output = linear.forward(input);
    cout << "output: " << endl;
    output.print();

    cout << "input: " << endl;
    input.print();

    /*matrix<int> A(3, 3);
    A.rand_init();
    A.print();*/
    
    /*matrix<int> X(1, 3);
    X.rand_init();
    cout << "matrix X:" << endl;
    X.print();

    matrix<int> W(3, 3);
    W.rand_init();
    cout << "matrix W:" << endl;
    W.print();

    matrix<int> B(1, 3);
    B.rand_init();
    cout << "matrix B:" << endl;
    B.print();

    matrix<int> output = X * W + B;
    cout << "matrix output:" << endl;
    output.print();*/
    return 0;
}