//%%writefile ejemplo.cu

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

int main() {
    const int N = 1 << 20; //1M para prueba; cambiar a 32<<20 para algo mas realista pero costoso
    const int M = 8;

    thrust::default_random_engine rng(1337);
    thrust::uniform_int_distribution<int> dist;
    thrust::host_vector<int> h_vec(N);
    thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

    std::cout << "Antes (primeros " << M << "): ";
    for (int i = 0; i < M; ++i) std::cout << h_vec[i] << " ";
    std::cout << "\n";

    thrust::device_vector<int> d_vec = h_vec;
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.begin() + M, h_vec.begin());

    std::cout << "Despues (primeros " << M << "): ";
    for (int i = 0; i < M; ++i) std::cout << h_vec[i] << " ";
    std::cout << "\n";

    return 0;
}

