#include "mpi.h"
// #include "acutest.h"
#include <thread>
#include <barrier>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

void send_and_recv(int rank, int size) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);

  std::vector<double> array(10);
  
  if (rank == 0) {
    for (int i = 0; i < 10; i++) {
      array[i] = 20;
    }
    MPI_Send(array.data(), 10, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(array.data(), 10, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  MPI_Finalize();
  printf("MPI_Send / MPI_Recv: I am thread %d with value %f\n", rank, array[0]);
}

void bcast(int rank, int size) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);
  std::vector<double> array(10);
  if (rank == 0) {
    for (int i = 0; i < 10; i++) {
      array[i] = 10;
    }
  }
  MPI_Bcast(array.data(), 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  printf("MPI_Bcast: I am thread %d with value %f\n", rank, array[0]);
}

void cart(int rank, int size) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);

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
  MPI_Finalize();
}

void reduce(int rank, int size) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);

  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Reduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("MPI_Reduce: I am thread %d with value %f\n", rank, d_reduce[0]);
  }
  MPI_Finalize();
}

void allreduce(int rank, int size) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);

  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Allreduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("MPI_Allreduce: I am thread %d with value %f\n", rank, d_reduce[0]);
  MPI_Finalize();
}

void scan(int rank, int size) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);

  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Scan(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("MPI_Scan: I am thread %d with value %f\n", rank, d_reduce[0]);
  MPI_Finalize();
}
// void (*f)(int, int)
void run(int rank, int size) {
  send_and_recv(rank, size);
  MPI_Reset();
  bcast(rank, size);
  MPI_Reset();
  cart(rank, size);
  MPI_Reset();
  reduce(rank, size);
  MPI_Reset();
  allreduce(rank, size);
  MPI_Reset();
  scan(rank, size);
  MPI_Reset();
  
}

int main(int argc, char* argv[]) {
  int num_threads = 2;
  std::thread t1(run, 0, 2);
  std::thread t2(run, 1, 2);

  t1.join();
  t2.join();
}

// void test_initialize() {
//   int num_threads = 2;
//   std::thread t1(run, 0, 2);
//   std::thread t2(run, 1, 2);

//   t1.join();
//   t2.join();

//   MPI_Finalize();
// }

// TEST_LIST = {
//     { "initialize", test_initialize }
//     { NULL, NULL }
// };