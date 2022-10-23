#include <string>
#include <future>
#include <iostream>
#include <vector>

#include <condition_variable>
#include <iostream>

#include <chrono>
#include <thread>

#include "mpi.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
using namespace emscripten ;
#endif

void longCompute(int id, int iterations) {
    double val = 0 ;
    for (int i=0; i<10000; ++i) {
        for (int j=0; j<iterations; ++j) {
            val += 3.1415 ;
        }
    }
    std::cout << "  thread " << (id) << ": " << val << std::endl ;
}

void main_func(int rank, int size) {
    std::cout << "  Started " << rank << std::endl;
    MPI_Register_Thread(rank, size);
    MPI_Init(NULL, NULL);
    int initialized;
    longCompute(rank, 40000);

    MPI_Finalize();
    MPI_Reset();
    std::cout << "  MPI ended " << rank << std::endl;
}

void testMpi(int workload) {
    std::cout << "Using std::thread" << std::endl ;
    std::vector<std::thread> threads ;
    for (int i = 0; i < workload; i++) {
        threads.push_back( std::thread(main_func, i, workload) ) ;
    }
    std::cerr << "start threads"<< std::endl ;
    for (auto &th : threads) {
        th.join() ;
    }
    std::cerr << "end threads"<< std::endl ;
}

void testSerial(int workload) {
    std::cout << "Using serial" << std::endl ;
    for (int i = 0; i < workload; i++) {
        longCompute(i, 40000) ;
    }
    std::cerr << "end serial"<< std::endl ;
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(test) {
    function("testMpi", &testMpi);
    function("testSerial", &testSerial);
}
#else
int main() {
    testSerial(4);
    testMpi(4);
}
#endif
