#include "net.h"

#include <iostream>

using carml::Tensor;

// Constructor & Destructor
Net::Net(Configuration config) {
    // Set the configuration
    config = config;

    // Initialize Weights and Biases Vectors
    weights = std::vector<Tensor>(config.n_hidden + 1);
    biases = std::vector<Tensor>(config.n_hidden + 1);

    // Initialize Weights
    weights[0] = Tensor(config.i_size, config.h_size);
    for (int i = 1; i < config.n_hidden - 1; i++) {
        weights[i] = Tensor(config.h_size, config.h_size);
    }
    weights[config.n_hidden] = Tensor(config.h_size, config.o_size);

    // Initialize Biases
    biases[0] = Tensor(1, config.h_size);
    for (int i = 1; i < config.n_hidden - 1; i++) {
        biases[i] = Tensor(1, config.h_size);
    }
    biases[config.n_hidden] = Tensor(1, config.o_size);

    // Fill with Random Values
    for (int i = 0; i < config.n_hidden + 1; i++) {
        weights[i].fill_with_random(config.i_size);
        biases[i].fill_with_random(config.i_size);
    }
}

Net::~Net() {
    // Nothing to do here
}

// Forward Propagation
vector<float> Net::forward(Tensor input) {
    





    // Reset Utility Vectors
    hidden_layer = vector<float>(config.hidden_layer_size, 0.0);

    // Input Layer -> Hidden Layer
    for (int i = 0; i < config.hidden_layer_size; i++) {
        for (int j = 0; j < config.input_size; j++) {
            float dt = input[j] * i2h_weights[j][i] + i2h_b[i];
            if (dt*dt > 0) {
                hidden_layer[i] += dt;
            }
        }
    }

    vector<float> output_layer(config.output_size, 0.0);
    float sum = 0.0;

    for (int i = 0; i < config.output_size; i++) {
        for (int j = 0; j < config.hidden_layer_size; j++) {
            output_layer[i] += hidden_layer[j] * h2o_weights[j][i] + h2o_b[i];
            sum += output_layer[i];
        }
    }

    // Normalize Output Layer
    for (int i = 0; i < config.output_size; i++) {
        output_layer[i] /= sum;
    }

    // Return the normalized output layer
    return output_layer;
}

void softmax(vector<float>& output_layer) {
    float max_output = *max_element(output_layer.begin(), output_layer.end());
    float sum = 0.0;
    
    for(int i = 0; i < output_layer.size(); i++) {
        output_layer[i] = exp(output_layer[i] - max_output);  // Subtract max for numerical stability
        sum += output_layer[i];
    }
    
    for(int i = 0; i < output_layer.size(); i++) {
        output_layer[i] /= sum;
    }
}

float cross_entropy_loss(const vector<float>& target, const vector<float>& output_layer) {
    float loss = 0.0;
    for(int i = 0; i < target.size(); i++) {
        loss -= target[i] * log(output_layer[i] + 1e-8);  // Small constant for numerical stability
    }
    return loss;
}

// Backward Propagation
float Net::backward(vector<float> input, int correct_label) {
    vector<float> target(config.output_size, 0.0);
    target[correct_label] = 1.0;

    // Forward Propagation
    auto output_layer = forward(input);

    // Perform a Softmax on the output layer
    softmax(output_layer);

    output_error = vector<float>(config.output_size, 0.0);
    hidden_error = vector<float>(config.hidden_layer_size, 0.0);
    hidden_delta = vector<float>(config.hidden_layer_size, 0.0);
    output_delta = vector<float>(config.output_size, 0.0);

    // Output Layer -> Hidden Layer
    for(int i = 0; i < config.output_size; i++) {
        output_error[i] = output_layer[i] - target[i];
    }

    for (int i = 0; i < config.hidden_layer_size; i++) {
        for (int j = 0; j < config.output_size; j++) {
            output_delta[j] += hidden_layer[i] * output_error[j];
        }
    }

    // Hidden Layer -> Input Layer
    for (int i = 0; i < config.hidden_layer_size; i++) {
        for (int j = 0; j < config.output_size; j++) {
            hidden_error[i] += output_delta[j] * h2o_weights[i][j];
        }
    }

    for (int i = 0; i < config.hidden_layer_size; i++) {
        hidden_delta[i] = hidden_error[i] * (hidden_layer[i] > 0 ? 1 : 0);
    }

    // Update Weights
    for (int i = 0; i < config.hidden_layer_size; i++) {
        if (hidden_layer[i] <= 0) {
            continue;
        }
        for (int j = 0; j < config.output_size; j++) {
            float gradient = hidden_layer[i] * output_delta[j];
            // Clip the gradient
            if (gradient > config.clip_threshold) {
                gradient = config.clip_threshold;
            }
            if (gradient < -config.clip_threshold) {
                gradient = -config.clip_threshold;
            }
            // Update weight
            h2o_weights[i][j] -= config.lr * gradient;
        }
    }

    for (int i = 0; i < config.input_size; i++) {
        if (input[i] <= 0) {
            continue;
        }
        for (int j = 0; j < config.hidden_layer_size; j++) {
            float gradient = input[i] * hidden_delta[j];
            // Clip the gradient
            if (gradient > config.clip_threshold) {
                gradient = config.clip_threshold;
            }
            if (gradient < -config.clip_threshold) {
                gradient = -config.clip_threshold;
            }
            // Update weight
            i2h_weights[i][j] -= config.lr * gradient;
        }
    }

    return cross_entropy_loss(target, output_layer);
}

// Training Function
void Net::train(matrix inputs, vector<int> labels, int epochs) {
    for (int i = 0; i < epochs; i++) {
        float avg_loss = 0.0;
        std::cout << "Epoch: " << i << "  .  .  .  " << std::endl;
        for (int j = 0; j < inputs.size(); j++) {
            avg_loss += backward(inputs[j], labels[j]);
            normalize_weights();
        }
        std::cout << "Average Loss: " << avg_loss / inputs.size() << std::endl;
    }
}

// Testing Function
void Net::test(matrix inputs, vector<int> labels) {
    int num_right = 0;
    for (int i = 0; i < inputs.size(); i++) {
        vector<float> output = forward(inputs[i]);
        int max_index = 0;
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > output[max_index]) {
                max_index = j;
            }
        }
        //std::cout << "Prediction: " << max_index << std::endl;
        if (max_index == labels[i]) {
            num_right++;
        }
    }
    std::cout << "\nPerformance: " << num_right << "/" << inputs.size() << "  ||  " << double(num_right) / double(inputs.size()) << std::endl;
}

// Single use inference function
void Net::inference(vector<float> input, int label) {
    vector<float> output = forward(input);
    std::cout << std::endl;
    for (int j = 0; j < output.size(); j++) {
        std::cout << output[j] << " ";
    }
    std::cout << std::endl;

    // Find the highest output activation
    int max_index = 0;
    for (int i = 0; i < output.size(); i++) {
        if (output[i] > output[max_index]) {
            max_index = i;
        }
    }
    std::cout << "Predicted Label: " << max_index << std::endl;
}

void Net::normalize_weights () {// Weight Normalization
    float sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < config.hidden_layer_size; i++) {
        sum1 += h2o_weights[i][0];
    }

    // Finish the sum
    sum1 = 0;
    for (int i = 0; i < config.hidden_layer_size; i++) {
        for (int j = 0; j < config.output_size; j++) {
            sum1 += h2o_weights[i][j];
        }
    }

    sum2 = 0;
    for (int i = 0; i < config.input_size; i++) {
        for (int j = 0; j < config.hidden_layer_size; j++) {
            sum2 += i2h_weights[i][j];
        }
    }

    // Normalize the weights
    for (int i = 0; i < config.hidden_layer_size; i++) {
        for (int j = 0; j < config.output_size; j++) {
            h2o_weights[i][j] /= sum1;
        }
    }

    for (int i = 0; i < config.input_size; i++) {
        for (int j = 0; j < config.hidden_layer_size; j++) {
            i2h_weights[i][j] /= sum2;
        }
    }
}