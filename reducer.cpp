#include <mpi.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <omp.h>

int rank, n_ranks;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis(0.0, 1.0);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    std::cout << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout <<  omp_get_num_threads() << " openmp threads" << std::endl;

    long n = 1000000;

    {    
        double** stuff = (double**)malloc(1000 * sizeof(double*));
        for (int i = 0; i < 1000; i++) {
            stuff[i] = (double*)malloc(1000 * sizeof(double));
            for (int j = 0; j < 1000; j++) {
                stuff[i][j] = dis(gen);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 1000; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                MPI_Reduce(MPI_IN_PLACE, stuff[i], 1000, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            } else {
                MPI_Reduce(stuff[i], stuff[i], 1000, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        auto now = std::chrono::high_resolution_clock::now();
        if (rank == 0) std::cout << (now - start).count() / 1e9 << std::endl;
    }

    {
        std::vector<double> stuff;
        stuff.reserve(n);
        for (long i = 0; i < n; i++) {
            stuff.push_back(dis(gen));
        }

        MPI_Barrier(MPI_COMM_WORLD);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            std::cout << "reducing " << stuff.size() << " doubles" << std::endl;
            MPI_Reduce(MPI_IN_PLACE, stuff.data(), stuff.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(stuff.data(), stuff.data(), stuff.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);

        auto now = std::chrono::high_resolution_clock::now();
        if (rank == 0) std::cout << (now - start).count() / 1e9 << std::endl;

    } 

    MPI_Finalize();
}
