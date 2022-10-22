/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU Genesize_datatyperal Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* Single-processor "stub" versions of MPI routines */

#include "mpi.h"

// #include "../version.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <barrier>
#include <thread>
#include <iostream>
#include <tuple>


#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

/* data structure for double/int */

struct _mpi_double_int {
  double value;
  int proc;
};
typedef struct _mpi_double_int double_int;

/* extra MPI_Datatypes registered by MPI_Type_contiguous */

#define MAXEXTRA_DATATYPE 16

int nextra_datatype;
MPI_Datatype *ptr_datatype[MAXEXTRA_DATATYPE];
int index_datatype[MAXEXTRA_DATATYPE];
size_t size_datatype[MAXEXTRA_DATATYPE];

static std::map<pthread_t,int> thread_id_map;
static std::map<int,bool> finalized_map;
static std::map<int,bool> initialized_map;

std::barrier barrier(2);

// std::function 
MPI_STATE::MPI_STATE(int num_threads) : 
  barrier(std::barrier(num_threads)) 
{
  // send_barriers
  for (int i = 0; i < num_threads; i++) {
    for (int j = 0; j < num_threads; j++) {
      auto barrier = std::make_shared<std::barrier<>>(2);
      send_barriers[std::make_pair(i, j)] = barrier;
    }
  }
}

MPI_STATE state(2);

// static int _num_threads = 2;
// MPI_STATE &get_state(int num_threads = _num_threads)
// {
//   _num_threads = num_threads;
  
//   /* Initializes bar the first time through this function */
//   static MPI_STATE state(num_threads);

//   return state;
// }

/* ---------------------------------------------------------------------- */
/* MPI Functions */
/* ---------------------------------------------------------------------- */

int get_rank() {
  return thread_id_map[pthread_self()];
}

int get_size() {
  return thread_id_map.size();
}

static std::atomic<int64_t> initialize_counter = 0;

void MPI_Register_Thread(int rank) {
  while (initialize_counter < rank) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  thread_id_map[pthread_self()] = rank;
  initialize_counter+=1;
  if (rank == 0) {
    size_datatype[MPI_INT] = sizeof(int);
    size_datatype[MPI_FLOAT] = sizeof(float);
    size_datatype[MPI_DOUBLE] = sizeof(double);
    size_datatype[MPI_CHAR] = sizeof(char);
    size_datatype[MPI_BYTE] = sizeof(char);
    size_datatype[MPI_LONG] = sizeof(long);
    size_datatype[MPI_LONG_LONG] = sizeof(long long);
    size_datatype[MPI_DOUBLE_INT] = sizeof(double)+sizeof(int);
  }

  barrier.arrive_and_wait();
}

int MPI_Init(int *argc, char ***argv)
{
  if (thread_id_map.count(pthread_self()) == 0) {
    printf("MPI_Init called without having called MPI_Register_Thread");
    return 1;
  }

  initialized_map[get_rank()] = true;
  printf("Initialized MPI with rank %d\n", get_rank());

  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Initialized(int *flag)
{
  pthread_t thread_id = pthread_self();
  *flag = (thread_id_map.count(thread_id) > 0) ? 1 : 0;
  printf("MPI_Initialized\n");
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Finalized(int *flag)
{
  *flag = (finalized_map.count(get_rank()) < 0) ? 1 : 0;
  printf("MPI_Finalized\n");
  return 0;
}

/* ---------------------------------------------------------------------- */

/* return "LAMMPS MPI STUBS" as name of the library */

int MPI_Get_library_version(char *version, int *resultlen)
{
  const char string[] = "WASM MPI";
  printf("MPI_Get_library_version\n");
  int len;

  if (!version || !resultlen) return MPI_ERR_ARG;

  len = strlen(string);
  memcpy(version, string, len + 1);
  *resultlen = len;
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

/* return "localhost" as name of the processor */

int MPI_Get_processor_name(char *name, int *resultlen)
{
  const char host[] = "wasm-mpi";
  int len;

  if (!name || !resultlen) return MPI_ERR_ARG;

  len = strlen(host);
  memcpy(name, host, len + 1);
  *resultlen = len;
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

/* return MPI version level. v1.2 is not 100% correct, but close enough */

int MPI_Get_version(int *major, int *minor)
{
  printf("MPI_Get_version\n");
  if (!major || !minor) return MPI_ERR_ARG;

  *major = 1;
  *minor = 2;
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_rank(MPI_Comm comm, int *me)
{
  pthread_t thread_id = pthread_self();

  *me = thread_id_map[thread_id];
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_size(MPI_Comm comm, int *nprocs)
{
  pthread_t thread_id = pthread_self();
  
  *nprocs = thread_id_map.count(thread_id);
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Abort(MPI_Comm comm, int errorcode)
{
  printf("MPI_Abort called.\n");
  exit(1);
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Finalize()
{
  if (initialized_map.count(get_rank()) == 0) {
    printf("MPI WARNING: MPI not yet initialized\n");
    return 1;
  }
  if (finalized_map.count(get_rank()) == 1) {
    printf("MPI WARNING: MPI already finalized\n");
    return 1;
  }
  finalized_map[get_rank()] = true;
  return 0;
}

/* ---------------------------------------------------------------------- */

double MPI_Wtime()
{
  printf("MPI_Wtime\n");
#if defined(_WIN32)
  double t;

  t = GetTickCount();
  t /= 1000.0;
  return t;
#else
  double time;
  struct timeval tv;

  gettimeofday(&tv, NULL);
  time = 1.0 * tv.tv_sec + 1.0e-6 * tv.tv_usec;
  return time;
#endif
}

/* ---------------------------------------------------------------------- */

/* include sizes of user defined datatypes, stored in extra lists */

static int stubtypesize(MPI_Datatype datatype)
{
  if (datatype == MPI_INT)
    return sizeof(int);
  else if (datatype == MPI_FLOAT)
    return sizeof(float);
  else if (datatype == MPI_DOUBLE)
    return sizeof(double);
  else if (datatype == MPI_CHAR)
    return sizeof(char);
  else if (datatype == MPI_BYTE)
    return sizeof(char);
  else if (datatype == MPI_LONG)
    return sizeof(long);
  else if (datatype == MPI_LONG_LONG)
    return sizeof(uint64_t);
  else if (datatype == MPI_DOUBLE_INT)
    return sizeof(double_int);
  else {
    int i;
    for (i = 0; i < nextra_datatype; i++)
      if (datatype == index_datatype[i]) return size_datatype[i];
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Type_size(MPI_Datatype datatype, int *size)
{
  printf("MPI_Type_size\n");
  if (size == NULL) return MPI_ERR_ARG;

  *size = stubtypesize(datatype);
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Request_free(MPI_Request *request)
{
  printf("MPI_Request_free\n");
  return 0;
}

/* ---------------------------------------------------------------------- */
std::map<std::pair<int,int>, const void*> mpi_send_map;
std::map<std::pair<int,int>, int> mpi_send_size_map;
std::map<std::pair<int,int>, MPI_Datatype> mpi_send_datatype_map;
// std::map<std::pair<int,int>, std::barrier> mpi_send_barrier;

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  auto threads_pair = std::make_pair(get_rank(), dest);
  
  mpi_send_map[threads_pair] = buf;
  mpi_send_size_map[threads_pair] = count;
  mpi_send_datatype_map[threads_pair] = datatype;
  state.send_barriers[threads_pair]->arrive_and_wait();
  state.send_barriers[threads_pair]->arrive_and_wait();
  
  mpi_send_map.erase(threads_pair);
  mpi_send_size_map.erase(threads_pair);
  mpi_send_datatype_map.erase(threads_pair);

  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
              MPI_Request *request)
{
  printf("MPI_Isend\n");
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI Stub WARNING: Should not send message to self\n");
    ++callcount;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  printf("MPI_Rsend\n");
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI Stub WARNING: Should not rsend message to self\n");
    ++callcount;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
             MPI_Status *status)
{
  auto threads_pair = std::make_pair(source, get_rank());
  
  state.send_barriers[threads_pair]->arrive_and_wait();
  auto buffer_size = mpi_send_size_map[threads_pair] * sizeof(mpi_send_datatype_map[threads_pair]);
  std::memcpy(buf, mpi_send_map[threads_pair], buffer_size);
  state.send_barriers[threads_pair]->arrive_and_wait();

  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
              MPI_Request *request)
{
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI WARNING: MPI_Irecv not implemented\n");
    ++callcount;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
  static int callcount = 0;
  if (callcount == 0) {
        printf("MPI WARNING: MPI_Wait not implemented\n");

    ++callcount;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Waitall(int n, MPI_Request *request, MPI_Status *status)
{
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI WARNING: MPI_Waitall not implemented\n");
    ++callcount;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Waitany(int count, MPI_Request *request, int *index, MPI_Status *status)
{
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI WARNING: MPI_Waitany not implemented\n");
    ++callcount;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Sendrecv(const void *sbuf, int scount, MPI_Datatype sdatatype, int dest, int stag,
                 void *rbuf, int rcount, MPI_Datatype rdatatype, int source, int rtag,
                 MPI_Comm comm, MPI_Status *status)
{
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI WARNING: MPI_Sendrecv not implemented\n");
    ++callcount;
  }

  // if (source == get_rank()) {
  //   MPI_Send(sbuf, scount, sdatatype, dest, stag, comm);
  //   MPI_Recv(rbuf, rcount, rdatatype, source, rtag, comm, status);
  // } else {
  //   MPI_Recv(rbuf, rcount, rdatatype, source, rtag, comm, status);
  //   MPI_Send(sbuf, scount, sdatatype, dest, stag, comm);
  // }

  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count)
{
  printf("MPI_Get_count\n");
  static int callcount = 0;
  if (callcount == 0) {
    printf("MPI Stub WARNING: Should not get count of message to self\n");
    ++callcount;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *comm_out)
{
  *comm_out = comm + 1;
  printf("MPI_Comm_split\n");
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *comm_out)
{
  printf("MPI_Comm_dup\n");
  *comm_out = comm + 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_free(MPI_Comm *comm)
{
  printf("MPI_Comm_free\n");
  return 0;
}

/* ---------------------------------------------------------------------- */

MPI_Fint MPI_Comm_c2f(MPI_Comm comm)
{
  printf("MPI_Comm_c2f\n");
  return comm;
};

/* ---------------------------------------------------------------------- */

MPI_Comm MPI_Comm_f2c(MPI_Fint comm)
{
  printf("MPI_Comm_f2c\n");
  return comm;
};

/* ---------------------------------------------------------------------- */

int MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
{
  printf("MPI_Comm_group\n");
  *group = comm;
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm)
{
  printf("MPI_Comm_create\n");
  *newcomm = group;
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup)
{
  printf("MPI_Group_incl\n");
  if (n > 0)
    *newgroup = MPI_COMM_WORLD;
  else
    *newgroup = group;
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Group_free(MPI_Group *group)
{
  printf("MPI_Group_free\n");
  if (group) *group = MPI_GROUP_NULL;
  return 0;
}

/* ---------------------------------------------------------------------- */
std::map<int, std::tuple<int, int, int>> rank_cart_map;
std::map<std::tuple<int, int, int>, int> rank_cart_inverse_map;
std::tuple<int, int, int> cart_dims;

int MPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, int *periods, int reorder,
                    MPI_Comm *comm_cart)
{
  assert(ndims == 3 && "MPI_Cart_create only supports 3 dimensions.");
  assert(periods[0] == 1 && "MPI_Cart_create only supports 3 dimensions.");
  assert(periods[1] == 1 && "MPI_Cart_create only supports 3 dimensions.");
  assert(periods[2] == 1 && "MPI_Cart_create only supports 3 dimensions.");

  if (get_rank() == 0) {
    for (int i = 0; i < dims[0]; i++) {
      for (int j = 0; j < dims[1]; j++) {
        for (int k = 0; j < dims[2]; j++) {
          int thread_index = k + j * dims[2] + i * dims[2] * dims[1];
          rank_cart_map[thread_index] = std::make_tuple(i, j, k);
          rank_cart_inverse_map[rank_cart_map[thread_index]] = thread_index;
        }
      }
    }
    cart_dims = std::make_tuple(dims[0], dims[1], dims[2]);
  }
  barrier.arrive_and_wait();

  *comm_cart = comm_old;
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

int MPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords)
{
  dims[0] = dims[1] = dims[2] = 1;
  periods[0] = periods[1] = periods[2] = 1;
  auto my_coords = rank_cart_map[get_rank()];
  coords[0] = std::get<0>(my_coords);
  coords[1] = std::get<1>(my_coords);
  coords[2] = std::get<2>(my_coords);
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

int MPI_Cart_shift(MPI_Comm comm, int direction, int displ, int *source, int *dest)
{
  auto coords = rank_cart_map[*source];
  if (direction == 0) {
    int element = (std::get<0>(coords) + displ + std::get<0>(cart_dims)) % std::get<0>(cart_dims);
    std::get<0>(coords) = element;
  } else if (direction == 1) {
    int element = (std::get<1>(coords) + displ + std::get<1>(cart_dims)) % std::get<1>(cart_dims);
    std::get<1>(coords) = element;
  } else if (direction == 2) {
    int element = (std::get<2>(coords) + displ + std::get<2>(cart_dims)) % std::get<2>(cart_dims);
    std::get<2>(coords) = element;
  }
  *dest = rank_cart_inverse_map[coords];

  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank)
{
  *rank = rank_cart_inverse_map[std::make_tuple(coords[0], coords[1], coords[2])];
  return 0;
}

/* ---------------------------------------------------------------------- */

/* store size of user datatype in extra lists */

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype)
{
  printf("MPI_Type_contiguous\n");
  if (nextra_datatype == MAXEXTRA_DATATYPE) return -1;
  ptr_datatype[nextra_datatype] = newtype;
  index_datatype[nextra_datatype] = -(nextra_datatype + 1);
  size_datatype[nextra_datatype] = count * stubtypesize(oldtype);
  nextra_datatype++;
  return 0;
}

/* ---------------------------------------------------------------------- */

/* set value of user datatype to internal negative index,
   based on match of ptr */

int MPI_Type_commit(MPI_Datatype *datatype)
{
  printf("MPI_Type_commit\n");
  int i;
  for (i = 0; i < nextra_datatype; i++)
    if (datatype == ptr_datatype[i]) *datatype = index_datatype[i];
  return 0;
}

/* ---------------------------------------------------------------------- */

/* remove user datatype from extra lists */

int MPI_Type_free(MPI_Datatype *datatype)
{
  int i;
  printf("MPI_Type_free\n");
  for (i = 0; i < nextra_datatype; i++)
    if (datatype == ptr_datatype[i]) {
      ptr_datatype[i] = ptr_datatype[nextra_datatype - 1];
      index_datatype[i] = index_datatype[nextra_datatype - 1];
      size_datatype[i] = size_datatype[nextra_datatype - 1];
      nextra_datatype--;
      break;
    }
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op)
{
  printf("MPI_Op_create\n");
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Op_free(MPI_Op *op)
{
  printf("MPI_Op_free\n");
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Barrier(MPI_Comm comm)
{
  barrier.arrive_and_wait();
  printf("MPI_Barrier\n");
  return 0;
}

/* ---------------------------------------------------------------------- */
void *bcast_buffer = nullptr;
int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
  if (get_rank() == root) {
    bcast_buffer = buf;
    barrier.arrive_and_wait();
    barrier.arrive_and_wait();
    bcast_buffer = nullptr;
  } else {
    barrier.arrive_and_wait();
    memcpy(buf, bcast_buffer, count * sizeof(datatype));
    barrier.arrive_and_wait();
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                  MPI_Comm comm)
{
  printf("MPI_Allreduce\n");
  int n = count * stubtypesize(datatype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

template <typename T> void reduce_op(T *dest, T *source, int count, MPI_Op op) {
  for (int i = 0; i < count; i++) {
    if (op == MPI_SUM) {
      dest[i] += source[i];
    }
    if (op == MPI_MAX) {
      dest[i] = std::max(dest[i], source[i]);
    }
    if (op == MPI_MIN) {
      dest[i] = std::min(dest[i], source[i]);
    }
  }
}

#include <vector>

std::vector<double> tmprecvbuffer;
int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm)
{
  if (get_size() == 1) {
    printf("MPI_Reduce not supported for 1 thread.");
    return 1;
  }

  if (root == get_rank()) {
    assert(op != MPI_MAXLOC && "MPI_Reduce does not support MPI_MAXLOC.");
    assert(op != MPI_MINLOC && "MPI_Reduce does not support MPI_MINLOC.");
    assert(op != MPI_LOR && "MPI_Reduce does not support MPI_LOR.");
    assert(datatype != MPI_BYTE && "MPI_Reduce does not support MPI_BYTE.");

    if (count > tmprecvbuffer.size()) {
      tmprecvbuffer.resize(5 * count);
    }
    
    memcpy(recvbuf, sendbuf, count * size_datatype[datatype]);
    double *dblptr = reinterpret_cast<double *>(sendbuf);
    printf("Value starts at %f\n", dblptr[0]);
    dblptr = reinterpret_cast<double *>(recvbuf);
    printf("Value after initial copy %f\n", dblptr[0]);

    for (int i = 0; i < get_size(); i++) {
      if (i == root) {
        continue;
      }
      MPI_Recv(tmprecvbuffer.data(), count, datatype, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (datatype == MPI_INT) {
        reduce_op(reinterpret_cast<int*>(recvbuf), reinterpret_cast<int*>(tmprecvbuffer.data()), count, op);
      }
      if (datatype == MPI_FLOAT) {
        reduce_op(reinterpret_cast<float*>(recvbuf), reinterpret_cast<float*>(tmprecvbuffer.data()), count, op);
      }
      if (datatype == MPI_DOUBLE) {
        reduce_op(reinterpret_cast<double*>(recvbuf), reinterpret_cast<double*>(tmprecvbuffer.data()), count, op);
        printf("And now %f\n", dblptr[0]);
      }
      if (datatype == MPI_CHAR) {
        reduce_op(reinterpret_cast<char*>(recvbuf), reinterpret_cast<char*>(tmprecvbuffer.data()), count, op);
      }
      if (datatype == MPI_BYTE) {
        printf("MPI_SUM of MPI_BYTE is not supported");
      }
      if (datatype == MPI_LONG) {
        reduce_op(reinterpret_cast<long*>(recvbuf), reinterpret_cast<long*>(tmprecvbuffer.data()), count, op);
      }
      if (datatype == MPI_LONG_LONG) {
        reduce_op(reinterpret_cast<long long*>(recvbuf), reinterpret_cast<long long*>(tmprecvbuffer.data()), count, op);
      }
      if (datatype == MPI_DOUBLE_INT) {
        printf("MPI_SUM of MPI_DOUBLE_INT is not supported");
      }
    }
  } else {
    MPI_Send(sendbuf, count, datatype, 0, 0, MPI_COMM_WORLD);
  }
  
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
             MPI_Comm comm)
{
  printf("MPI_Scan\n");
  int n = count * stubtypesize(datatype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm)
{
  printf("MPI_Allgather\n");
  int n = sendcount * stubtypesize(sendtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm)
{
  printf("MPI_Allgatherv\n");
  int n = sendcount * stubtypesize(sendtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm)
{
  printf("MPI_Reduce_scatter\n");
  int n = *recvcounts * stubtypesize(datatype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
               MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  printf("MPI_Gather\n");
  int n = sendcount * stubtypesize(sendtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  printf("MPI_Gatherv\n");
  int n = sendcount * stubtypesize(sendtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  printf("MPI_Scatter\n");
  int n = recvcount * stubtypesize(recvtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
  printf("MPI_Scatterv\n");
  int n = recvcount * stubtypesize(recvtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, MPI_Comm comm)
{
  printf("MPI_Alltoall\n");
  int n = sendcount * stubtypesize(sendtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
                  void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
  printf("MPI_Alltoallv\n");
  int n = *sendcounts * stubtypesize(sendtype);

  if (sendbuf == MPI_IN_PLACE || recvbuf == MPI_IN_PLACE) return 0;
  memcpy(recvbuf, sendbuf, n);
  return 0;
}

/* ---------------------------------------------------------------------- */
