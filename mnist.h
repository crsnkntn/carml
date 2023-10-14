#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <ios>

typedef unsigned char uchar;

uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size);

uchar* read_mnist_labels(std::string full_path, int& number_of_labels);

int reverseInt (int i);

class mnist_dataset {
    public:
        std::vector<std::vector<float>> images;
        std::vector<int> labels;

        mnist_dataset(std::string images_path, std::string labels_path);

        ~mnist_dataset();

        void print_dataset_info();

        void print_labelled_image(int index);

        int n_images = 0;
        int im_size = 0;
        int n_labels = 0;

};