#pragma once

#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>
#include <typeinfo>

template<typename T>
void print_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            typename T::value_type value = array[i * n + j];
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void fill_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            array[i * n + j] = (i % 2)*3 + j;
        }
    }
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int m, int n) {
    thrust::host_vector<T> host_array(m*n);
    for(int i = 0; i < m * n; i++) {
        host_array[i] = (T)rand()/(T)RAND_MAX;
    }
    array = host_array;
}

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
    srand(0);
    thrust::host_vector<int> host_labels(n);
    for(int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}