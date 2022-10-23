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

double f( double a )
{
    return (4.0 / (1.0 + a*a));
}

void longCompute(int id, int iterations) {
    double val = 0 ;
    for (int i=0; i<10000; ++i) {
        for (int j=0; j<iterations; ++j) {
            val += 3.1415 ;
        }
    }
    std::cout << "  thread " << (id) << ": " << val << std::endl ;
}

void calc_pi_mpi(int rank, int size) {
    MPI_Register_Thread(rank, size);
    int done = 0, n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double startwtime = 0.0, endwtime;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);

    fprintf(stderr,"Process %d of %d on %s\n",
	    myid, numprocs, processor_name);

    n = 0;
    while (!done) {
        if (myid == 0) {
            if (n==0) {
                n=1024*numprocs; 
            } else {
                n=0;
            } 

            startwtime = MPI_Wtime();
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n == 0) {
            done = 1;
        } else {
            h   = 1.0 / (double) n;
            sum = 0.0;
            for (i = myid + 1; i <= n; i += numprocs) {
                x = h * ((double)i - 0.5);
                sum += f(x);
            }
            mypi = h * sum;
            MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (myid == 0) {
                printf("pi is approximately %.16f, Error is %.16f\n",
                       pi, fabs(pi - PI25DT));
		            endwtime = MPI_Wtime();
		        printf("wall clock time = %f\n",
		            endwtime-startwtime);
	        }
        }
    }
    MPI_Finalize();
    MPI_Reset();
}

void testMpi(int workload) {
    std::cout << "Using std::thread" << std::endl ;
    std::vector<std::thread> threads ;
    for (int i = 0; i < workload; i++) {
        threads.push_back( std::thread(calc_pi_mpi, i, workload) ) ;
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
