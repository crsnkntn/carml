#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <ios>
#include "mnist.h"

typedef unsigned char uchar;

uchar** read_mnist_images(std::string full_path, int& n_images, int& im_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&n_images, sizeof(n_images)), n_images = reverseInt(n_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        im_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[n_images];
        for(int i = 0; i < n_images; i++) {
            _dataset[i] = new uchar[im_size];
            file.read((char *)_dataset[i], im_size);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

uchar* read_mnist_labels(std::string full_path, int& n_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&n_labels, sizeof(n_labels)), n_labels = reverseInt(n_labels);

        uchar* _dataset = new uchar[n_labels];
        for(int i = 0; i < n_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

mnist_dataset::mnist_dataset(std::string images_path, std::string labels_path) {
    uchar** images_bits = read_mnist_images(images_path, n_images, im_size);
    uchar* labels_bits = read_mnist_labels(labels_path, n_labels);

    // Populate the images vector
    for(int i = 0; i < n_images; i++) {
        std::vector<float> image;
        for(int j = 0; j < im_size; j++) {
            image.push_back(images_bits[i][j] / 255.0);
        }
        images.push_back(image);
    }

    // Populate the labels vector
    for(int i = 0; i < n_labels; i++) {
        labels.push_back((int)labels_bits[i]);
    }
}

mnist_dataset::~mnist_dataset() {}

void mnist_dataset::print_dataset_info() {
    std::cout << "Number of images: " << n_images << std::endl;
    std::cout << "Image size: (" << im_size << ", " << im_size << std::endl;
}

void mnist_dataset::print_labelled_image(int index) {
    std::cout << "\n--\nLabel: " << (int)labels[index] << std::endl;
    for(int i = 0; i < im_size; i++) {
        if(i % 28 == 0) {
            std::cout << std::endl;
        }
        if(images[index][i] > 0) {
            std::cout << "#";
        } else {
            std::cout << " ";
        }
    }
}
