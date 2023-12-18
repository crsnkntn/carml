#include <iostream>
#include "lib/Tensor.hpp"


int main (int argc, char** argv) {
    int rank = 2;
    int* shape1 = new int[rank]{5, 300};
    int* shape2 = new int[rank]{300, 5};
    std::cout << "is it here A" << std::endl;
    carml::Tensor a(rank, shape1);
    std::cout << "is it here BN" << std::endl;
    carml::Tensor b(rank, shape2);

    int rand_seed = std::atoi(argv[1]);
    std::cout << "is it here C" << std::endl;
    a.fill_with_random(rand_seed);
    b.fill_with_random(rand_seed);

    carml::Tensor c = a.matmul(b);

    c.print();

    delete[] shape1;
    delete[] shape2;
    return 0;
}