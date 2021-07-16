#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include <math.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "helpers.h"

#include "../matplotlibcpp.h"
#include <vector>

namespace plt = matplotlibcpp;

__device__ float
squared_l2_distance(int d, float* point, float* centroid) {
  float dist = 0;
  for (int i=0; i<d; ++i)
    dist += pow((point[i] - centroid[i]), 2);
  return dist;
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.
__global__ void assign_clusters(int data_size,
                                int d,
                                int k,
                                const thrust::device_ptr<float> data,
                                const thrust::device_ptr<float> means,
                                thrust::device_ptr<float> new_sums,
                                thrust::device_ptr<int> counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= data_size) return;

  // Make global loads once.
  float* point = thrust::raw_pointer_cast(data + index * d);

  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {
    float* centroid = thrust::raw_pointer_cast(means + cluster * d);
    const float distance =
        squared_l2_distance(d, point, centroid);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  for (int i=0; i<d; ++i)
    atomicAdd(thrust::raw_pointer_cast(new_sums + best_cluster * d + i), point[i]);
  atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(int d,
                                  thrust::device_ptr<float> means,
                                  const thrust::device_ptr<float> new_sums,
                                  const thrust::device_ptr<int> counts) {
  const int cluster = threadIdx.x;
  const int count = max(1, counts[cluster]);
  
  for (int i=0; i<d; ++i)
    means[cluster * d + i] = new_sums[cluster * d + i] / count;
}

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

  thrust::device_vector<float> d_data(n * d);

  random_data(d_data, n, d);
  thrust::device_vector<float> d_mean(d_data.begin(), d_data.begin() + k * d);

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
  
  // thrust::device_vector<float> d_data(dataset, dataset+n*d);
  // thrust::device_vector<float> d_mean(centers, centers+k*d);

  // std::vector<float> x;
  // std::vector<float> y;

  // for (size_t pos = 0; pos < n; ++pos) {
  //   x.push_back(d_data[pos*2]);
  //   y.push_back(d_data[pos*2+1]);
  // }

  // plt::plot(x, y,  {{"color", "blue"}, {"marker", "."}, {"linestyle", ""}});

  thrust::device_vector<float> d_sums(k * d);
  thrust::device_vector<int> d_counts(k, 0);

  const int threads = 1024;
  const int blocks = (n + threads - 1) / threads;

  const auto start = std::chrono::high_resolution_clock::now();
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    thrust::fill(d_sums.begin(), d_sums.end(), 0);
    thrust::fill(d_counts.begin(), d_counts.end(), 0);

    assign_clusters<<<blocks, threads>>>(n, d, k,
                                         d_data.data(),
                                         d_mean.data(),
                                         d_sums.data(),
                                         d_counts.data());
    cudaDeviceSynchronize();

    compute_new_means<<<1, k>>>(d,
                                d_mean.data(),
                                d_sums.data(),
                                d_counts.data());
    cudaDeviceSynchronize();
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
  std::cerr << "Took: " << duration.count() << "s" << std::endl;

  for (size_t cluster = 0; cluster < k; ++cluster) {
    for (size_t i = 0; i < d; ++i){
      size_t idx = cluster * d + i;
      std::cout << d_mean[idx] << " ";
    }
    std::cout << std::endl;
  }

  // x.clear();
  // y.clear();

  // for (size_t c = 0; c < k; ++c) {
  //   x.push_back(d_mean[c*2]);
  //   y.push_back(d_mean[c*2+1]);
  // }

  // plt::plot(x, y,  {{"color", "red"}, {"marker", "o"}, {"linestyle", ""}});
  // plt::show();
}
