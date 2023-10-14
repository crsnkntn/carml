#pragma once
#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;

using matrix = vector<vector<float>>;

class Net {
    public:
        struct Configuration {
            int input_size;
            int output_size;
            int n_hidden_layers;
            int hidden_layer_size;
            
            float lr = 0.1;
            float clip_threshold = 1.0;
        };

    private:
        Configuration config;

        // Weight Matrices
        matrix i2h_weights;
        matrix h2o_weights;
        vector<float> i2h_b;
        vector<float> h2o_b;

        // Utility Vectors
        vector<float> hidden_layer;
        vector<float> hidden_error;
        vector<float> hidden_delta;

        vector<float> output_error;
        vector<float> output_delta;

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
        // - Returns the Loss
        float backward(vector<float> input, int correct_label);

        // Inference Function
        void inference(vector<float> input, int label);

        // Training Function
        void train(matrix inputs, vector<int> labels, int epochs);
        
        // Testing Function
        void test(matrix inputs, vector<int> labels);

        // Prints the Configuration
        void info();

    private:
        
        void normalize_weights ();
};