// Tensor with multiple dimensions
class tensor {
private:
    float* data;
    int* shape;
    int ndim;

public:
    tensor(float* data, int* shape, int ndim) : data(data), shape(shape), ndim(ndim) {}

    int size() {
        int s = 1;
        for (int i = 0; i < ndim; i++) {
            s *= shape[i];
        }
        return s;
    }

    float get(int* indices) {
        int index = 0;
        for (int i = 0; i < ndim; i++) {
            index = index * shape[i] + indices[i];
        }
        return data[index];
    }

    void set(int* indices, float value) {
        int index = 0;
        for (int i = 0; i < ndim; i++) {
            index = index * shape[i] + indices[i];
        }
        data[index] = value;
    }
};
