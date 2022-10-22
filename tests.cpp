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

void send_recv() {
  // int size;
  // MPI_Comm_size(MPI_COMM_WORLD, &size);
  // int my_rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // int buffer_send = (my_rank == 0) ? 12345 : 67890;
  // int buffer_recv;
  // int tag_send = 0;
  // int tag_recv = tag_send;
  // int peer = (my_rank == 0) ? 1 : 0;

  // // Issue the send + receive at the same time
  // printf("MPI process %d sends value %d to MPI process %d.\n", my_rank, buffer_send, peer);
  // MPI_Sendrecv(&buffer_send, 1, MPI_INT, peer, tag_send,
  //               &buffer_recv, 1, MPI_INT, peer, tag_recv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // printf("MPI process %d received value %d from MPI process %d.\n", my_rank, buffer_recv, peer);
}

void send_recv(int rank, int size) {
  // printf("I am %d and will do send and receive\n", rank);
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

void run(int rank, int size) {
#ifdef THREADS_MPI
  MPI_Register_Thread(rank);
  MPI_Init(NULL, NULL);
#endif

  send_recv(rank, size);
  bcast(rank, size);
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