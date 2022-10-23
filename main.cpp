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

std::vector<int> s = {40000, 40000, 40000, 40000} ;

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
    MPI_Register_Thread(rank);
    MPI_Init(NULL, NULL);
    int initialized;
    MPI_Initialized(&initialized);
    std::cout << "Initialized: " << initialized << std::endl;

    longCompute(rank, s[rank]);

    MPI_Finalize();
    // MPI_Reset();
}

void testMpi() {
    std::cout << "Using MPI" << std::endl ;

    std::vector<std::thread> threads ;
    for (int i = 0; i < s.size(); i++) {
        threads.push_back( std::thread(main_func, i, 2) );
    }
    std::cerr << "Starting MPI processes"<< std::endl ;
    for (auto &th : threads) {
        th.join() ;
    }
    std::cerr << "end MPI"<< std::endl ;
}

void testSerial() {
    std::cout << "Using serial" << std::endl ;
    for (int i = 0; i < s.size(); i++) {
        longCompute(i, s[i]) ;
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
    testSerial();
    testMpi();
}
#endif
