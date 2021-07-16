/*
Copyright 2013  Bryan Catanzaro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <thrust/device_vector.h>
#include "kmeans.h"
#include "timer.h"
#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include "helpers.h"

#include "../matplotlibcpp.h"
#include <vector>

namespace plt = matplotlibcpp;

// template<typename T>
// void print_array(T& array, int m, int n) {
//     for(int i = 0; i < m; i++) {
//         for(int j = 0; j < n; j++) {
//             typename T::value_type value = array[i * n + j];
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// template<typename T>
// void fill_array(T& array, int m, int n) {
//     for(int i = 0; i < m; i++) {
//         for(int j = 0; j < n; j++) {
//             array[i * n + j] = (i % 2)*3 + j;
//         }
//     }
// }

// template<typename T>
// void random_data(thrust::device_vector<T>& array, int m, int n) {
//     thrust::host_vector<T> host_array(m*n);
//     for(int i = 0; i < m * n; i++) {
//         host_array[i] = (T)rand()/(T)RAND_MAX;
//     }
//     array = host_array;
// }

// void random_labels(thrust::device_vector<int>& labels, int n, int k) {
//     thrust::host_vector<int> host_labels(n);
//     for(int i = 0; i < n; i++) {
//         host_labels[i] = rand() % k;
//     }
//     labels = host_labels;
// }

int main(int argc, const char* argv[]) {
    // if (argc != 5) {
    //   std::cerr << "usage: executable <number of data points> <number of cluster> <data dimension> <iteration>"
    //             << std::endl;
    //   std::exit(EXIT_FAILURE);
    // }

    // const int n = std::atoi(argv[1]);
    // const int k = std::atoi(argv[2]);
    // const int d = std::atoi(argv[3]);
    // const int iterations = std::atoi(argv[4]);

    int iterations = 50;
    int n = 1e6;
    int d = 64;
    int k = 128;

    thrust::device_vector<float> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<float> centroids(k * d);
    thrust::device_vector<float> distances(n);
    
    std::cout << "Generating random data" << std::endl;
    std::cout << "Number of points: " << n << std::endl;
    std::cout << "Number of dimensions: " << d << std::endl;
    std::cout << "Number of clusters: " << k << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    // std::cout << "Precision: " << typeid(T).name() << std::endl;
    
    random_data(data, n, d);
    random_labels(labels, n, k);

    // float dataset[] = {
    //   0.5, 0.5,
    //   1.5, 0.5,
    //   1.5, 1.5,
    //   0.5, 1.5,
    //   1.1, 1.2,
    //   0.5, 15.5,
    //   1.5, 15.5,
    //   1.5, 16.5,
    //   0.5, 16.5,
    //   1.2, 16.1,
    //   15.5, 15.5,
    //   16.5, 15.5,
    //   16.5, 16.5,
    //   15.5, 16.5,
    //   15.6, 16.2,
    //   15.5, 0.5,
    //   16.5, 0.5,
    //   16.5, 1.5,
    //   15.5, 1.5,
    //   15.7, 1.6};
    // float centers[] = {
    //   0.5, 0.5,
    //   1.5, 0.5,
    //   1.5, 1.5,
    //   0.5, 1.5};
     
    //   int iterations = 3;
    //   int n = 20;
    //   int d = 2;
    //   int k = 4;
    
    // thrust::device_vector<float> data(dataset, dataset+n*d);
    // thrust::device_vector<int> labels(n);
    // thrust::device_vector<float> centroids(centers, centers+k*d);
    // thrust::device_vector<float> distances(n);

    // std::vector<float> x;
    // std::vector<float> y;

    // for (size_t pos = 0; pos < n; ++pos) {
    //   x.push_back(data[pos*2]);
    //   y.push_back(data[pos*2+1]);
    // }

    // plt::plot(x, y,  {{"color", "blue"}, {"marker", "."}, {"linestyle", ""}});


    kmeans::timer t;
    t.start();
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances); // no false normally
    float time = t.stop();
    std::cout << "  Time: " << time/1000.0 << " s" << std::endl;
    
    std::cout << "==================== Centroids =================" << std::endl;
    for (size_t cluster = 0; cluster < k; ++cluster) {
      for (size_t i = 0; i < d; ++i){
        size_t idx = cluster * d + i;
        std::cout << centroids[idx] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "==================== Centroids =================" << std::endl;
    
    // x.clear();
    // y.clear();

    // for (size_t c = 0; c < k; ++c) {
    //   x.push_back(centroids[c*2]);
    //   y.push_back(centroids[c*2+1]);
    // }

    // plt::plot(x, y,  {{"color", "red"}, {"marker", "o"}, {"linestyle", ""}});
    // plt::show();

}
