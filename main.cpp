#include "mnist.h"
#include "mnist.cpp"
#include "net.h"
#include "net.cpp"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>

#define print(x) std::cout << x << std::endl;


int main(int argc, char** argv) {
    // set precision to 3 decimal places
    std::cout << std::fixed << std::setprecision(6);

    mnist_dataset dataset("mnist_data/train-images.idx3-ubyte", "mnist_data/train-labels.idx1-ubyte");
    
    // Get the training and validation datasets
    int train_set_size = 1000;
    matrix train_set(dataset.images.begin(), dataset.images.begin() + train_set_size);
    vector<int> train_labels(dataset.labels.begin(), dataset.labels.begin() + train_set_size);

    int test_set_size = 1000;
    matrix test_set(dataset.images.begin() + train_set_size, dataset.images.begin() + train_set_size + test_set_size);
    vector<int> test_labels(dataset.labels.begin() + train_set_size, dataset.labels.begin() + train_set_size + test_set_size);

    // Create the network
    float learning_rate = 0.5;
    float clip_threshold = 1.0;

    int n_hidden_layers = 2;
    int hidden_layer_size = 50;
    int n_classes = 10;

    Net net({
        dataset.im_size, 
        n_classes, 
        n_hidden_layers, 
        hidden_layer_size, 
        learning_rate, 
        clip_threshold
    });


    // Run Training Routine
    net.test(test_set, test_labels);
    for (int i = 0; i < 100; i++) {
        net.train(train_set, train_labels, 1);

        // Shuffle the dataset
        std::shuffle(train_set.begin(), train_set.end(), std::default_random_engine(0));
    }

    net.test(test_set, test_labels);

    return 0;
}
