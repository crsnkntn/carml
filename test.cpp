#include <iostream>
#include "lib/Tensor.h"


int main () {
    int rank = 2;
    int* shape1 = new int[rank]{300, 300};
    int* shape2 = new int[rank]{300, 10000};
    carml::Tensor a(rank, shape1);
    carml::Tensor b(rank, shape2);
    a.fill_with_random();
    b.fill_with_random();

    carml::Tensor c = carml::Tensor::matmul(a, b);

    //c.print();

    delete[] shape1;
    delete[] shape2;
    return 0;
}