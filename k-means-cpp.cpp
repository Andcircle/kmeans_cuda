#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>
#include <cstdlib>

using Point = std::vector<float>;
using DataFrame = std::vector<Point>;

float square(float value) {
  return value * value;
}

float squared_l2_distance(Point first, Point second) {
  float rslt = 0;
  for (size_t i=0; i< first.size(); ++i)
    rslt += square(first[i] - second[i]);
  return rslt;
}

DataFrame k_means(const DataFrame& data,
                  size_t d,
                  size_t k,
                  size_t number_of_iterations) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(seed());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  // Pick centroids as random points from the dataset.
  DataFrame means(k, Point(d));
  for (auto& cluster : means) {
    cluster = data[indices(random_number_generator)];
  }

  std::vector<size_t> assignments(data.size());
  for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {

    const auto start = std::chrono::high_resolution_clock::now();
    // Find assignments.
    for (size_t point = 0; point < data.size(); ++point) {
      auto best_distance = std::numeric_limits<float>::max();
      size_t best_cluster = 0;
      for (size_t cluster = 0; cluster < k; ++cluster) {
        const float distance =
            squared_l2_distance(data[point], means[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      assignments[point] = best_cluster;
    }

    // Sum up and count points for each cluster.
    DataFrame new_means(k, Point(d));
    std::vector<size_t> counts(k, 0);
    for (size_t point = 0; point < data.size(); ++point) {
      const auto cluster = assignments[point];
      for (size_t i=0; i<d; ++i)
        new_means[cluster][i] += data[point][i];
      counts[cluster] += 1;
    }

    // Divide sums by counts to get new centroids.
    for (size_t cluster = 0; cluster < k; ++cluster) {
      // Turn 0/0 into 0/1 to avoid zero division.
      const auto count = std::max<size_t>(1, counts[cluster]);
      for (size_t i=0; i<d; ++i)
        means[cluster][i] = new_means[cluster][i] / count;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cerr << "Iteration:" << iteration << " took: " << duration.count() << "s" << std::endl;
  }

  return means;
}

int main(int argc, const char* argv[]) {
  size_t iterations = 50;
  size_t n = 20000;
  size_t d = 768;
  size_t k = 8;

  // size_t iterations = 10;
  // size_t n = 20;
  // size_t d = 2;
  // size_t k = 4;

  DataFrame data(n, Point(d));
  for(auto& p: data)
    for(auto& x: p)
      x = (float)rand()/(float)RAND_MAX;

  DataFrame means(k, Point(d));

  const auto start = std::chrono::high_resolution_clock::now();
  means = k_means(data, d, k, iterations);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto duration =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

  std::cerr << "Took: " << duration.count() << "s" << std::endl;

  for (auto& mean : means) {
    for (auto& x: mean)
      std::cout << x << " ";
    std::cout<<std::endl;
  }
}
