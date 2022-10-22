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

std::mutex mutex_;
std::condition_variable condVar; 

bool dataReady{false};

void waitingForWork(){
    std::cout << "Waiting " << std::endl;
    std::unique_lock<std::mutex> lck(mutex_);
    condVar.wait(lck, []{ return dataReady; });   // (4)
    std::cout << "Running " << std::endl;
}

void setDataReady(){
    {
        std::lock_guard<std::mutex> lck(mutex_);
        dataReady = true;
    }
    std::cout << "Data prepared" << std::endl;
    condVar.notify_one();                        // (3)
}

void mutex() {
    std::thread t1(waitingForWork);               // (1)
    std::thread t2(setDataReady);                 // (2)

    t1.join();
    t2.join();
}

void main_func(int rank) {
    
    MPI_Register_Thread(rank);
    // Initialize the MPI environment
    std::this_thread::sleep_for(std::chrono::milliseconds(2000 * rank + 1));
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double *array = new double[10];
    memset(array, 10, sizeof(double));
    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            array[i] = 10;
        }
    }
    MPI_Bcast(array, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("MPI_Bcast: I am thread %d with value %f\n", rank, array[0]);

    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            array[i] = 20;
        }
        MPI_Send(array, 10, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(array, 10, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("MPI_Send / MPI_Recv: I am thread %d with value %f\n", rank, array[0]);

    if (rank == 0) {
        for (int i = 0; i < 10; i++) {
            array[i] = 20;
        }
    }

    // MPI_Sendrecv(array, )

    // printf("MPI_SendRecv: I am thread %d with value %f\n", rank, array[0]);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}

void go() {
    std::thread t1(main_func, 0);
    std::thread t2(main_func, 1);

    t1.join();
    t2.join();
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(test) {
    function("go", &go) ;
    function("mutex", &mutex);
}
#else
int main() {
    go();
}
#endif
