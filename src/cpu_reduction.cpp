#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace chrono;

int main()
{
    const size_t N = 1 << 24;  // 2^24 elements

    vector<float> data(N);

    // Generate input data
    for (size_t i = 0; i < N; i++)
    {
        data[i] = static_cast<float>((i % 100) + 1) / 100.0f;
    }

    // CPU reduction
    double sum = 0.0;

    auto start = high_resolution_clock::now();

    for (size_t i = 0; i < N; i++)
    {
        sum += data[i];
    }

    auto stop = high_resolution_clock::now();

    double time_ms =
        duration<double, milli>(stop - start).count();

    cout << fixed << setprecision(6);

    cout << "CPU Reduction Results\n";
    cout << "---------------------\n";
    cout << "Input elements: " << N << endl;
    cout << "Reference sum: " << sum << endl;
    cout << "CPU execution time: " << time_ms << " ms" << endl;

    return 0;
}
