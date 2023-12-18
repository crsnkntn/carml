#ifndef CARML_TENSOR_H
#define CARML_TENSOR_H

#include <iostream>
#include <cassert>
#include <random>
#include <iomanip>
#include <memory>

namespace carml {
    class Tensor {
        protected:
            int _rank = 0;
            int _size = 0;
            int* _shape = nullptr;
            float* _data = nullptr;

        public:
            Tensor(int rank, int* shape, float* data);
            Tensor(int rank, int* shape);
            Tensor(int m, int n);
            ~Tensor();

            void fill_with_zeros();
            void fill_with_ones();
            void fill_with_random(int n);
            void fill_with_value(float v);
            void apply(std::function<float(float)> func);

            int rank () const;
            int size () const;
            int* shape () const;
            float* data () const;
            int shape (int index) const;

            void print() const;

            Tensor& matmul (const Tensor& b);
            Tensor& batchmul (const Tensor& b);
            Tensor T() const;

            float operator[](int index) const;
            Tensor& operator+(const float a) const;
            Tensor& operator-(const float a) const;
            Tensor& operator*(const float a) const;
            Tensor& operator/(const float a) const;
            Tensor& operator+(const Tensor& other) const;
            Tensor& operator-(const Tensor& other) const;
            Tensor& operator*(const Tensor& other) const;
            Tensor& operator/(const Tensor& other) const;

        private:
            Tensor& unary_helper (const float a, 
                std::function<float(float, float)> func) const;
            Tensor& binary_helper (const Tensor& other, 
                std::function<float(float, float)> func) const;
    };

}

#endif // CARML_TENSOR_H