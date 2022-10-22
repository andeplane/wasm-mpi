#ifdef THREADS_MPI
#include "mpi.h"
#include <thread>
#include<barrier>
#else
#include <mpi.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <vector>

void send_and_recv(int rank, int size) {
  std::vector<double> array(10);
  
  if (rank == 0) {
    for (int i = 0; i < 10; i++) {
      array[i] = 20;
    }
    MPI_Send(array.data(), 10, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(array.data(), 10, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  printf("MPI_Send / MPI_Recv: I am thread %d with value %f\n", rank, array[0]);
}

void bcast(int rank, int size) {
  std::vector<double> array(10);
  if (rank == 0) {
    for (int i = 0; i < 10; i++) {
      array[i] = 10;
    }
  }
  MPI_Bcast(array.data(), 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
  printf("MPI_Bcast: I am thread %d with value %f\n", rank, array[0]);
}

void cart(int rank, int size) {
  int periods[3];
  int reorder = 0;
  periods[0] = periods[1] = periods[2] = 1;
  MPI_Comm cartesian;
  int procgrid[3];
  procgrid[0] = 2;
  procgrid[1] = 1;
  procgrid[2] = 1;

  int myloc[3];

  MPI_Cart_create(MPI_COMM_WORLD,3,procgrid,periods,reorder,&cartesian);
  MPI_Cart_get(cartesian,3,procgrid,periods,myloc);
  
  printf("MPI_Cart: I am thread %d with location %d %d %d\n", rank, myloc[0], myloc[1], myloc[2]);
  int source = 0;
  int destination = 0;
  MPI_Cart_shift(cartesian,0,1, &source, &destination);
  printf("Shifted for 0, dim 0, displ 1: %d\n", destination);
  MPI_Cart_shift(cartesian,0,-1, &source, &destination);
  printf("Shifted for 0, dim 0, displ -1: %d\n", destination);
  
  MPI_Cart_shift(cartesian,1,1, &source, &destination);
  printf("Shifted for 0, dim 1, displ 1: %d\n", destination);
  MPI_Cart_shift(cartesian,1,-1, &source, &destination);
  printf("Shifted for 0, dim 1, displ -1: %d\n", destination);
  
}

void run(int rank, int size) {
#ifdef THREADS_MPI
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);
#endif

  send_and_recv(rank, size);
  bcast(rank, size);
  cart(rank, size);
}

int main(int argc, char* argv[]) {
#ifdef THREADS_MPI
  int num_threads = 2;
  // auto initialize_barrier = std::barrier(num_threads);
  std::thread t1(run, 0, 2);
  std::thread t2(run, 1, 2);

  t1.join();
  t2.join();
#else
  MPI_Init(&argc, &argv);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(size != 2)
  {
      printf("%d MPI processes used, please use 2.\n", size);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Prepare parameters
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  run(my_rank);
#endif

  MPI_Finalize();
}