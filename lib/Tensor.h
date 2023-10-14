#pragma once
#include <iostream>
#include <cassert>
#include <random>
#include <iomanip>
#include <memory>
#include "Tensor.h"

namespace carml {
    class Tensor {
        protected:
            int _rank;
            int _size;
            int* _shape;
            float* _data;

        public:
            Tensor(int rank, int* shape, float* data) 
            : _rank(rank) {
                // Create a copy of the shape
                _size = 1;
                for (int dim = 0; dim < _rank; dim++) {
                    _size *= shape[dim];
                }
                
                _shape = new int[_rank];
                _data = new float[_size];
                memcpy(_shape, shape, _rank * sizeof(int));
                memcpy(_data, data, _size * sizeof(float));

                print();
            }

            Tensor(int rank, int* shape) 
            : _rank(rank) {
                // Create a copy of the shape
                _size = 1;
                for (int dim = 0; dim < _rank; dim++) {
                    _size *= shape[dim];
                }

                _shape = new int[_rank];
                memcpy(_shape, shape, _rank * sizeof(int));

                _data = new float[_size];
                for (int i = 0; i < _size; i++) {
                    _data[i] = 0.0;
                }
            }

            Tensor(int m, int n) {
                _rank = 2;
                _size = m * n;
                _shape = new int[_rank]{m, n};
                _data = new float[_size];
                for (int i = 0; i < _size; i++) {
                    _data[i] = 0.0;
                }
            }

            ~Tensor() {
                delete[] _shape;
                delete[] _data;
            }

            void fill_with_random(int n) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, std::sqrt(2.0 / n));

                for (int i = 0; i < _size; i++) {
                    _data[i] = dis(gen);
                }
                std::cout << "Filled with random\n";
            }

            int* shape () {
                return _shape;
            }

            int rank () const {
                return _rank;
            }

            // Getter Functions
            int shape (int index) const {
                if (index > _rank) {
                    throw std::invalid_argument("Index out of bounds");
                }
                return _shape[index];
            }
            
            // Data Getter
            float operator[](int index) const {
                if (index > _size) {
                    throw std::invalid_argument("Index out of bounds");
                }
                return _data[index];
            }

            // Print a Tensor, aesthetically
            void print() const {
                std::cout << std::setprecision(2);
                std::cout << "Tensor of rank " << _rank << "\n";
                std::cout << "Shape: [";
                for (int i = 0; i < _rank; i++) {
                    std::cout << _shape[i];
                    if (i != _rank - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]\n";
                std::cout << "Data: [\n";
                for (int i = 0; i < _size; i++) {
                    if (i != 0 && i % _shape[0] == 0)
                        std::cout << "\n" << _data[i];
                    else
                        std::cout << _data[i] << ", \t\t";
                }
                std::cout << "]\n";
            }

            void apply(std::function<float(float)> func) {
                for (int i = 0; i < _shape[0]; i++) {
                    _data[i] = func(_data[i]);
                }
            }

            Tensor& operator_helper_scaler (const float a, std::function<float(float, float)> func) const {
                // Create new Tensor Data
                float* data = new float[_size];

                // Operate on the tensor elements, add to new tensor
                for (int i = 0; i < _size; i++) {
                    data[i] = func(_data[i], a);
                }

                // Create a copy of the shape
                int* shape_ptr = new int[_rank];
                for (int dim = 0; dim < _rank; dim++) {
                    shape_ptr[dim] = _shape[dim];
                }

                return *(new Tensor(_rank, shape_ptr, data));
            }

            Tensor& operator_helper_tensor (const Tensor& other, std::function<float(float, float)> func) const {
                // Check if the tensors are compatible
                assert(_rank == other.rank());
                for (int dim = 0; dim < _rank; dim++) {
                    assert(_shape[dim] == other.shape(dim));
                }

                // Create new Tensor Data
                float* data = new float[_size];

                // Operate on the tensor elements, add to new tensor
                for (int i = 0; i < _size; i++) {
                    data[i] = func(_data[i], other[i]);
                    std::cout << data[i] << "\n";
                }

                // Create a copy of the shape
                int* shape_ptr = new int[_rank];
                for (int dim = 0; dim < _rank; dim++) {
                    shape_ptr[dim] = _shape[dim];
                    std::cout << shape_ptr[dim] << "\n";
                }

                return *(new Tensor(_rank, shape_ptr, data));
            }

            Tensor& operator_helper_matmul (const Tensor& other) {
                // Check if the tensors are compatible
                assert(_rank == 2);
                assert(_rank == other.rank());
                assert(_shape[1] == other.shape(0));

                // Create new Tensor Data
                float* data = new float[_shape[0] * other.shape(1)];

                // Create a copy of the shape
                int* shape_ptr = new int[2]{_shape[0], other.shape(1)};

                // Multiply row by column
                for (int row = 0; row < _shape[0]; row++) {
                    for (int col = 0; col < other.shape(1); col++) {
                        float sum = 0.0;
                        for (int i = 0; i < _shape[1]; i++) {
                            sum += _data[row * _shape[1] + i] * other[i * other.shape(1) + col];
                        }
                        data[row * other.shape(1) + col] = sum;
                    }
                }

                return *(new Tensor(_rank, shape_ptr, data));
            }

            Tensor& operator_helper_batchmul (const Tensor& other) {
                // Check that the tensors are compatible
                assert(_rank == 3);
                assert(other.rank() == 2);

                // Create new Tensor Data
                data = new float[_shape[0] * _shape[1] * other.shape(1)];

                shape = new int[3]{_shape[0], _shape[1], other.shape(1)};

                // Multiply row by column
                for (int batch = 0; batch < _shape[0]; batch++) {
                    for (int row = 0; row < _shape[1]; row++) {
                        for (int col = 0; col < other.shape(1); col++) {
                            float sum = 0.0;
                            for (int i = 0; i < _shape[2]; i++) {
                                sum += _data[batch * _shape[1] * _shape[2] + row * _shape[2] + i] * other[i * other.shape(1) + col];
                            }
                            data[batch * _shape[1] * other.shape(1) + row * other.shape(1) + col] = sum;
                        }
                    }
                }

                return *(new Tensor(_rank, shape, data));
            }

            static Tensor& matmul (Tensor& a, Tensor& b) {
                return a.operator_helper_matmul(b);
            }

            static Tensor& batchmul (Tensor& a, Tensor& b) {
                return a.operator_helper_batchmul(b);
            }
            // Transpose
            Tensor T() const {
                // Create new Tensor Data
                float* data = new float[_shape[0]];

                // Operate on the tensor elements, add to new tensor
                for (int i = 0; i < _shape[0]; i++) {
                    data[i] = _data[i];
                }

                // Create a copy of the shape
                int* shape_ptr = new int[_rank];
                for (int dim = 0; dim < _rank; dim++) {
                    shape_ptr[dim] = _shape[dim];
                }

                return Tensor(_rank, shape_ptr, data);
            }

            // 1. Addition
            Tensor& operator+(const float a) const {
                return operator_helper_scaler(a, [](float a, float b) {return a + b;});
            }

            Tensor& operator+(const Tensor& other) const {
                return operator_helper_tensor(other, [](float a, float b) {return a + b;});
            }

            // 2. Subtraction
            Tensor& operator-(const float a) const {
                return operator_helper_scaler(a, [](float a, float b) {return a - b;});
            }

            Tensor& operator-(const Tensor& other) const {
                return operator_helper_tensor(other, [](float a, float b) {return a - b;});
            }

            // 3. Multiplication
            Tensor& operator*(const float a) const {
                return operator_helper_scaler(a, [](float a, float b) {return a * b;});
            }

            Tensor& operator*(const Tensor& other) const {
                return operator_helper_tensor(other, [](float a, float b) {return a * b;});
            }
            
            // 4. Division
            Tensor& operator/(const float a) const {
                return operator_helper_scaler(a, [](float a, float b) {return a / b;});
            }

            Tensor& operator/(const Tensor& other) const {
                return operator_helper_tensor(other, [](float a, float b) {return a / b;});
            }
    };
}