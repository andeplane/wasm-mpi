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
      if (i < count) {
        TEST_CHECK(array[i] == i + count);
      } else {
        TEST_CHECK(array[i] == 15);
      }
    }
  }
}

void bcast(int rank, int size) {
  int count = 10;
  std::vector<double> array(2 * count, 15);
  if (rank == 0) {
    for (int i = 0; i < count; i++) {
      array[i] = i + count;
    }
  }
  MPI_Bcast(array.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for (int i = 0; i < 2 * count; i++) {
    if (i < count) {
      TEST_CHECK(array[i] == i + count);
    } else {
      TEST_CHECK(array[i] == 15);
    }
  }
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
  
  int source = 0;
  int destination = 0;
  MPI_Cart_shift(cartesian,0,1, &source, &destination);
  MPI_Cart_shift(cartesian,0,-1, &source, &destination);
  
  MPI_Cart_shift(cartesian,1,1, &source, &destination);
  MPI_Cart_shift(cartesian,1,-1, &source, &destination);
}

void reduce(int rank, int size) {
  int count = 10;
  std::vector<double> d_values(2 * count, 15);
  for (int i = 0; i < count; i++) {
    d_values[i] = (rank + 1) * i + count;
  }
  
  std::vector<double> d_reduce(count);
  int initialized; 
  MPI_Initialized(&initialized);
  MPI_Reduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    TEST_CHECK(d_reduce[0] == 20.000000);
    TEST_CHECK(d_reduce[1] == 23.000000);
    TEST_CHECK(d_reduce[2] == 26.000000);
    TEST_CHECK(d_reduce[3] == 29.000000);
    TEST_CHECK(d_reduce[4] == 32.000000);
    TEST_CHECK(d_reduce[5] == 35.000000);
    TEST_CHECK(d_reduce[6] == 38.000000);
    TEST_CHECK(d_reduce[7] == 41.000000);
    TEST_CHECK(d_reduce[8] == 44.000000);
    TEST_CHECK(d_reduce[9] == 47.000000);
  } else {
    for (int i = 0; i < 2 * count; i++) {
      if (i < count) {
        TEST_CHECK(d_values[i] == (rank + 1) * i + count);
      } else {
        TEST_CHECK(d_values[i] == 15);
      }
    }
    for (int i = 0; i < count; i++) {
      TEST_CHECK(d_reduce[i] == 0);
    }
  }

  MPI_Reduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, 1, MPI_COMM_WORLD);
  
  if (rank == 1) {
    TEST_CHECK(d_reduce[0] == 20.000000);
    TEST_CHECK(d_reduce[1] == 23.000000);
    TEST_CHECK(d_reduce[2] == 26.000000);
    TEST_CHECK(d_reduce[3] == 29.000000);
    TEST_CHECK(d_reduce[4] == 32.000000);
    TEST_CHECK(d_reduce[5] == 35.000000);
    TEST_CHECK(d_reduce[6] == 38.000000);
    TEST_CHECK(d_reduce[7] == 41.000000);
    TEST_CHECK(d_reduce[8] == 44.000000);
    TEST_CHECK(d_reduce[9] == 47.000000);
  }
}

void allreduce(int rank, int size) {
  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Allreduce(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < count; i++) {
    TEST_CHECK(d_reduce[i] == 20.000000);
  }
}

void scan(int rank, int size) {
  int count = 10;
  std::vector<double> d_values(count, 10);
  std::vector<double> d_reduce(count);
  MPI_Scan(d_values.data(), d_reduce.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  printf("MPI_Scan: I am thread %d with value %f\n", rank, d_reduce[0]);
}

void sendrecv(int rank, int size) {
  int count = 10;
  double value = 10 * (rank + 1);
  // double *send = new double[count];
  // for (int i = 0; i<count; i++) {
  //   send[i] = value;
  // }
  std::vector<double> send(count, value);
  std::vector<double> recv(count);

  int send_peer = (rank + 1) % size;
  int recv_peer = (rank - 1 + size) % size;
  printf("I am %d with ptr %p and will send %f to %d and receive from %d\n", rank, send.data(), value, send_peer, recv_peer);

  MPI_Sendrecv(send.data(), count, MPI_DOUBLE, send_peer, 0, recv.data(), count, MPI_DOUBLE, recv_peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  printf("I am thread %d and received %f from %d\n", rank, recv[0], recv_peer);
}

void run(int rank, int size, void (*f)(int, int)) {
  MPI_Register_Thread(rank, size);
  MPI_Init(NULL, NULL);

  f(rank, size);
  MPI_Finalize();

  MPI_Reset();
}

void test_send_and_recv() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, send_and_recv);
  std::thread t2(run, 1, 2, send_and_recv);

  t1.join();
  t2.join();
}

void test_bcast() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, bcast);
  std::thread t2(run, 1, 2, bcast);

  t1.join();
  t2.join();
}
void test_cart() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, cart);
  std::thread t2(run, 1, 2, cart);

  t1.join();
  t2.join();
}
void test_reduce() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, reduce);
  std::thread t2(run, 1, 2, reduce);

  t1.join();
  t2.join();
}
void test_allreduce() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, allreduce);
  std::thread t2(run, 1, 2, allreduce);

  t1.join();
  t2.join();
}
void test_scan() {
  int num_threads = 2;
  std::thread t1(run, 0, 2, scan);
  std::thread t2(run, 1, 2, scan);

  t1.join();
  t2.join();
}
void test_sendrecv() {
  int num_threads = 4;
  std::thread t1(run, 0, 4, sendrecv);
  std::thread t2(run, 1, 4, sendrecv);
  std::thread t3(run, 2, 4, sendrecv);
  std::thread t4(run, 3, 4, sendrecv);

  t1.join();
  t2.join();
  t3.join();
  t4.join();
}

TEST_LIST = {
    { "send_and_recv", test_send_and_recv },
    { "bcast", test_bcast },
    { "reduce", test_reduce },
    { "allreduce", test_allreduce },
    { "sendrecv", test_sendrecv},
    // { "scan", test_scan },
    { NULL, NULL }
};