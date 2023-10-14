#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "Tensor.h"

using std::string;
using std::vector;

using matrix = vector<vector<float>>;
/**
 * @brief 
 * Neural Network with H equally sized hidden layers
 * 
 */
class Net {
    public:
        struct Configuration {
            int i_size;
            int o_size;
            int n_hidden;
            int h_size;
            
            float lr = 0.1;
            float clip_threshold = 1.0;
        };

    private:
        /** Network Configuration
         * doesn't need to be here
         * doesn't change afer initialization
         */
        Configuration config;

        /** Weight Matrices
         * Size: n_hidden_layers + 1
        */
        std::vector<carml::Tensor> weights;

        /** Bias Vectors
         * Size: n_hidden_layers + 1
        */
        std::vector<carml::Tensor> biases;
        

    public:
        // Constructor & Destructor
        Net(Configuration config);
        ~Net();

        // Should Not Be Used
        Net(const Net&) = delete;
        Net& operator=(const Net&) = delete;

        // Forward Propagation
        vector<float> forward(vector<float> input);

        // Backward Propagation
        float backward(vector<float> input, int correct_label);

        // Inference Function
        void inference(vector<float> input, int label);

        // Training Function
        void train(matrix inputs, vector<int> labels, int epochs);
        
        // Testing Function
        void test(matrix inputs, vector<int> labels);

        // Prints the Configuration
        void config();

    private:
        
        void normalize_weights ();
};