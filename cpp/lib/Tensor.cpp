#include "Tensor.hpp"

using namespace carml;

carml::Tensor::Tensor(int rank, int* shape, float* data) : _rank(rank) {
     // Create a copy of the shape
    _size = 1;
    for (int dim = 0; dim < _rank; dim++) {
        _size *= shape[dim];
    }
    
    _shape = new int[_rank];
    _data = new float[_size];
    memcpy(_shape, shape, _rank * sizeof(int));
    memcpy(_data, data, _size * sizeof(float));
}

carml::Tensor::Tensor(int rank, int* shape) : _rank(rank) {
    // Create a copy of the shape
    _size = 1;
    for (int dim = 0; dim < _rank; dim++) {
        _size *= shape[dim];
    }

    _shape = new int[_rank];
    memcpy(_shape, shape, _rank * sizeof(int));

    _data = new float[_size];
    fill_with_zeros();
}

carml::Tensor::Tensor(int m, int n) : _rank(2) {
    _rank = 2;
    _size = m * n;
    _shape = new int[_rank]{m, n};
    _data = new float[_size];
    fill_with_zeros();
}

Tensor::~Tensor() {
    delete[] _shape;
    delete[] _data;
}

// Helper Functions
Tensor& Tensor::unary_helper(const float a, std::function<float(float, float)> func) const {
    float* data = new float[_size];

    for (int i = 0; i < _size; i++) {
        data[i] = func(_data[i], a);
    }

    int* shape_ptr = new int[_rank];
    memcpy(shape_ptr, _shape, _rank * sizeof(int));

    return *(new Tensor(_rank, shape_ptr, data));
}

Tensor& Tensor::binary_helper(const Tensor& other, std::function<float(float, float)> func) const {
    assert(_rank == other.rank());

    for (int dim = 0; dim < _rank; dim++) {
        assert(_shape[dim] == other.shape(dim));
    }

    float* data = new float[_size];

    for (int i = 0; i < _size; i++) {
        data[i] = func(_data[i], other[i]);
    }

    int* shape_ptr = new int[_rank];
    memcpy(shape_ptr, _shape, _rank * sizeof(int));

    return *(new Tensor(_rank, shape_ptr, data));
}

void dot_product_with_batch_offset (float* dataDest, float* dataA, float* dataB, 
    int* shapeA, int* shapeB, int batchOffset) {
        for (int row = 0; row < shapeA[0]; row++) {
            for (int col = 0; col < shapeB[1]; col++) {
                float sum = 0.0;
                for (int i = 0; i < shapeA[1]; i++) {
                    sum += dataA[batchOffset * shapeA[1] * shapeB[1] * row * shapeA[1] + i] * dataB[i * shapeB[1] + col];
                }
                dataDest[batchOffset * shapeA[1] * shapeB[1] + row * shapeB[1] + col] = sum;
            }
        }
}

// Matrix Multiplication
Tensor& Tensor::matmul(const Tensor& other) {
    assert(_rank == 2);
    assert(_rank == other.rank());
    assert(_shape[1] == other.shape(0));

    float* data = new float[_shape[0] * other.shape(1)];
    int* shape_ptr = new int[2]{_shape[0], other.shape(1)};

    dot_product_with_batch_offset(data, _data, other.data(), _shape, other.shape(), 0);

    return *(new Tensor(_rank, shape_ptr, data));
}

// Batch Matrix Multiplication
Tensor& Tensor::batchmul(const Tensor& other) {
    assert(_rank == 3);
    assert(other.rank() == 2);

    float* data = new float[_shape[0] * _shape[1] * other.shape(1)];
    int* shape = new int[3]{_shape[0], _shape[1], other.shape(1)};

    for (int batch = 0; batch < _shape[0]; batch++) {
        dot_product_with_batch_offset(data, _data, other.data(), _shape, other.shape(), batch);
    }

    return *(new Tensor(_rank, shape, data));
}

//Fill Functions
void Tensor::fill_with_value(float v) {
    for (int i = 0; i < _size; i++) {
        _data[i] = v;
    }
}

void Tensor::fill_with_zeros() {
    fill_with_value(0.0);
}

void Tensor::fill_with_ones() {
    fill_with_value(1.0);
}

void Tensor::fill_with_random(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, std::sqrt(2.0 / n));

    for (int i = 0; i < _size; i++) {
        _data[i] = dis(gen);
    }
}

void Tensor::apply(std::function<float(float)> func) {
    for (int i = 0; i < _size; i++) {
        _data[i] = func(_data[i]);
    }
}

// Getter Functions
int* Tensor::shape() const {
    return _shape;
}

int Tensor::rank() const {
    return _rank;
}

int Tensor::shape(int index) const {
    if (index >= _rank) {
        throw std::invalid_argument("Index out of bounds");
    }
    return _shape[index];
}

int Tensor::size() const {
    return _size;
}

float* Tensor::data() const {
    return _data;
}

// Data Getter
float Tensor::operator[](int index) const {
    if (index >= _size) {
        throw std::invalid_argument("Index out of bounds");
    }
    return _data[index];
}

void Tensor::print() const {
    std::cout << std::setprecision(2);
    std::cout << "Tensor of rank " << _rank << "\n";
    std::cout << "Shape: [";

    for (int dim = 0; dim < _rank; dim++) {
        std::cout << _shape[dim];
        if (dim != _rank - 1) {
            std::cout << ", ";
        }
    }

    std::cout << "]\nData: [\n";

    for (int i = 0; i < _size; i++) {
        std::cout << _data[i];
        if ((i + 1) % _shape[0] == 0) {
            std::cout << "\n";
        } else {
            std::cout << ",\t";
        }
    }

    std::cout << "]\n";
}

Tensor Tensor::T() const {
    float* data = new float[_shape[0]];

    for (int i = 0; i < _shape[0]; i++) {
        data[i] = _data[i];
    }

    int* shape_ptr = new int[_rank];
    memcpy(shape_ptr, _shape, _rank * sizeof(int));

    return Tensor(_rank, shape_ptr, data);
}

// Operator Overloads
Tensor& Tensor::operator+(const float a) const {
    return unary_helper(a, [](float a, float b) { return a + b; });
}

Tensor& Tensor::operator+(const Tensor& other) const {
    return binary_helper(other, [](float a, float b) { return a + b; });
}

Tensor& Tensor::operator-(const float a) const {
    return unary_helper(a, [](float a, float b) { return a - b; });
}

Tensor& Tensor::operator-(const Tensor& other) const {
    return binary_helper(other, [](float a, float b) { return a - b; });
}

Tensor& Tensor::operator*(const float a) const {
    return unary_helper(a, [](float a, float b) { return a * b; });
}

Tensor& Tensor::operator*(const Tensor& other) const {
    return binary_helper(other, [](float a, float b) { return a * b; });
}

Tensor& Tensor::operator/(const float a) const {
    return unary_helper(a, [](float a, float b) { return a / b; });
}

Tensor& Tensor::operator/(const Tensor& other) const {
    return binary_helper(other, [](float a, float b) { return a / b; });
}
