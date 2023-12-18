#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "Tensor.hpp"
#include "Shape.hpp"



namespace carml {
    /**
     * @brief 
     * Parameters is a useful tool from TensorFlow
     * Essentially its a wrapper around a Tensor
     * 
     */
    class Parameters : public Tensor {
        private:
            std::string _name;
            Shape _shape;
        public:
            void shape() {
                std::cout << "\"" << _name << "\" Shape: (";
                for (auto s = _shape.begin(); s < _shape.end() - 1; s++) {
                    std::cout << *s << ", ";
                }
                std::cout << *(_shape.end() - 1) << ")" << std::endl;
            }
    };
}