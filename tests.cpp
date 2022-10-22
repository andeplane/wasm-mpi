#include "mpi.h"
#include "acutest.h"
#include <thread>
#include <barrier>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

void send_and_recv(int rank, int size) {
  int count = 10;
  std::vector<double> array(2 * count, 15);
  
  if (rank == 0) {
    for (int i = 0; i < count; i++) {
      array[i] = i + count;
    }
    MPI_Send(array.data(), count, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(array.data(), count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
  
  if (rank == 1) {
    for (int i = 0; i < 2 * count; i++) {
      printf("array[%d] = %f\n", i, array[i]);
      if (i < count) {
        TEST_CHECK(array[i] == i + count);
      } else {
        TEST_CHECK(array[i] == 15);
      }
    }
  }
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

void reduce(int rank, int size) {
  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Reduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("MPI_Reduce: I am thread %d with value %f\n", rank, d_reduce[0]);
  }
}

void allreduce(int rank, int size) {
  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Allreduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("MPI_Allreduce: I am thread %d with value %f\n", rank, d_reduce[0]);
}

void scan(int rank, int size) {
  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Scan(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("MPI_Scan: I am thread %d with value %f\n", rank, d_reduce[0]);
}

void run(int rank, int size, void (*f)(int, int)) {
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);

  f(rank, size);
  MPI_Finalize();

  MPI_Reset();
}

void test_send_and_recv() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, &send_and_recv);
  std::thread t2(run, 1, 2, &send_and_recv);

  t1.join();
  t2.join();
}

void test_bcast() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, &bcast);
  std::thread t2(run, 1, 2, &bcast);

  t1.join();
  t2.join();
}
void test_cart() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, &cart);
  std::thread t2(run, 1, 2, &cart);

  t1.join();
  t2.join();
}
void test_reduce() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, &reduce);
  std::thread t2(run, 1, 2, &reduce);

  t1.join();
  t2.join();
}
void test_allreduce() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, &allreduce);
  std::thread t2(run, 1, 2, &allreduce);

  t1.join();
  t2.join();
}
void test_scan() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, &scan);
  std::thread t2(run, 1, 2, &scan);

  t1.join();
  t2.join();
}

TEST_LIST = {
    { "send_and_recv", test_send_and_recv },
    // { "bcast", test_bcast },
    // { "cart", test_cart },
    // { "reduce", test_reduce },
    // { "allreduce", test_allreduce },
    // { "scan", test_scan },
    { NULL, NULL }
};