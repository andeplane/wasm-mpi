#include "mpi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cassert>
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
static std::atomic<int64_t> initialize_counter = 0;

int get_rank() {
  return thread_id_map[pthread_self()];
}

int get_size() {
  return thread_id_map.size();
}

MPI_STATE::MPI_STATE(int max_threads) : 
  max_threads(max_threads)
{
  // send_barriers
  for (int i = 0; i < max_threads; i++) {
    for (int j = 0; j < max_threads; j++) {
      auto barrier = std::make_shared<std::barrier<>>(2);
      send_barriers[std::make_pair(i, j)] = barrier;
    }
  }
}
const int MAX_THREADS = 8;

MPI_STATE state(MAX_THREADS);

void MPI_STATE::setRequest(MPI_Request_type type, int source, int dest, MPI_Datatype datatype, int count, int tag, MPI_Request request) {
  MPI_RequestKey key = {
    datatype,
    type,
    count,
    source,
    dest,
    tag
  };
  std::lock_guard<std::mutex> guard(state.requestMutex);
  state.requests[key] = request;
}

std::optional<MPI_Request> MPI_STATE::getRequest(MPI_Request_type type, int source, int dest, MPI_Datatype datatype, int count, int tag) {
  MPI_RequestKey key = {
    datatype,
    type,
    count,
    source,
    dest,
    tag
  };
  std::lock_guard<std::mutex> guard(state.requestMutex);
  std::optional<MPI_Request> request;
  if (state.requests.count(key) > 0) {
    request = state.requests[key];
  }
  return request;
}

void MPI_STATE::removeRequest(MPI_Request request) {
  std::lock_guard<std::mutex> guard(state.requestMutex);
  MPI_RequestKey key = {
    request.datatype,
    request.type,
    request.count,
    request.source,
    request.dest,
    request.tag
  };

  state.requests.erase(key);
}

void MPI_Reset() {
  if (get_rank() == 0) {
    thread_id_map.clear();
    finalized_map.clear();
    initialized_map.clear();
    initialize_counter = 0;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/* ---------------------------------------------------------------------- */
/* MPI Functions */
/* ---------------------------------------------------------------------- */

void MPI_Register_Thread(int rank, int num_threads) {
  while (initialize_counter < rank) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  thread_id_map[pthread_self()] = rank;
  initialize_counter+=1;

  if (rank == 0) {
    state.num_threads = num_threads;
  }
  _MPI_Barrier(num_threads);
}

int MPI_Init(int *argc, char ***argv)
{
  if (thread_id_map.count(pthread_self()) == 0) {
    printf("MPI_Init called without having called MPI_Register_Thread");
    return MPI_ERR_OTHER;
  }
  if (get_rank() == 0) {
    initialize_counter = 0;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  while (initialize_counter < get_rank()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  initialized_map[get_rank()] = true;
  initialize_counter+=1;
  if (initialize_counter > thread_id_map.size()) {
    printf("MPI_Init called for more threads than registered using MPI_Register_Thread. Aborting!");
    return MPI_ERR_OTHER;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

int MPI_Initialized(int *flag)
{
  *flag = (initialized_map.count(get_rank()) > 0) ? 1 : 0;
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Finalized(int *flag)
{
  *flag = (finalized_map.count(get_rank()) < 0) ? 1 : 0;
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
  *me = get_rank();
  return 0;
}

/* ---------------------------------------------------------------------- */

int MPI_Comm_size(MPI_Comm comm, int *nprocs)
{
  *nprocs = state.num_threads;
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
  int initialized;
  MPI_Initialized(&initialized);
  if (initialized == 0) {
    printf("MPI WARNING: MPI not yet initialized\n");
    return 1;
  }
  if (finalized_map.count(get_rank()) == 1) {
    printf("MPI WARNING: MPI already finalized\n");
    return 1;
  }

  if (get_rank() == 0) {
    initialize_counter = 0;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  while (initialize_counter < get_rank()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  finalized_map[get_rank()] = true;
  initialize_counter++;

  return MPI_SUCCESS;
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
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{
  MPI_Request request = {
    datatype,
    MPI_Send_t,
    count,
    get_rank(),
    dest,
    tag,
    comm,
    buf,
    nullptr,
  };

  state.setRequest(MPI_Send_t, get_rank(), dest, datatype, count, tag, request);
  
  // Lock thread until data is read
  while (state.getRequest(MPI_Send_t, get_rank(), dest, datatype, count, tag)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  return MPI_SUCCESS;
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
  // Lock thread until data is read
  while (!state.getRequest(MPI_Send_t, source, get_rank(), datatype, count, tag)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::optional<MPI_Request> request = state.getRequest(MPI_Send_t, source, get_rank(), datatype, count, tag);
  std::memcpy(buf, request->sendbuf, request->count * stubtypesize(request->datatype));
  state.removeRequest(*request);

  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
              MPI_Request *request)
{
  request->count = count;
  request->datatype = datatype;
  request->source = source;
  request->tag = tag;
  request->comm = comm;
  request->type = MPI_Irecv_t;
  
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
  if (request->type == MPI_Irecv_t) {
    MPI_Recv(request->recvbuf, request->count, request->datatype, request->source, request->tag, request->comm, status);
  } else {
    printf("MPI WARNING: MPI_Wait for this request type not supported\n");
  }
  return MPI_SUCCESS;
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
std::mutex sendrecv_mutex;
std::map<std::pair<int, int>, const void *> sendrecv_sendbuffers;
std::map<std::pair<int, int>, int> sendrecv_sendcount;
std::map<std::pair<int, int>, int> sendrecv_senddatatype;

int MPI_Sendrecv(const void *sbuf, int scount, MPI_Datatype sdatatype, int dest, int stag,
                 void *rbuf, int rcount, MPI_Datatype rdatatype, int source, int rtag,
                 MPI_Comm comm, MPI_Status *status)
{
  // Note: This requires that all processors will send and receive, but this is not necessarily the case.
  {
    std::lock_guard<std::mutex> guard(sendrecv_mutex);
    sendrecv_sendbuffers[std::make_pair(get_rank(), dest)] = sbuf;
    sendrecv_sendcount[std::make_pair(get_rank(), dest)] = scount;
    sendrecv_senddatatype[std::make_pair(get_rank(), dest)] = sdatatype;
    double *dbuf = (double*)sbuf;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  {
    std::lock_guard<std::mutex> guard(sendrecv_mutex);
    memcpy(rbuf, 
      sendrecv_sendbuffers[std::make_pair(source, get_rank())],
      sendrecv_sendcount[std::make_pair(source, get_rank())] * stubtypesize(sendrecv_senddatatype[std::make_pair(source, get_rank())])
    );
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
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
  MPI_Barrier(MPI_COMM_WORLD);

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

void _MPI_Barrier(int num_threads)
{
  // TODO: Get rid of this insane hack.
  if (num_threads == 2) {
    state.barrier_2.arrive_and_wait();
  } else if (num_threads == 3) {
    state.barrier_3.arrive_and_wait();
  } else if (num_threads == 4) {
    state.barrier_4.arrive_and_wait();
  } else if (num_threads == 5) {
    state.barrier_5.arrive_and_wait();
  } else if (num_threads == 6) {
    state.barrier_6.arrive_and_wait();
  } else if (num_threads == 7) {
    state.barrier_7.arrive_and_wait();
  } else if (num_threads == 8) {
    state.barrier_8.arrive_and_wait();
  }
}

int MPI_Barrier(MPI_Comm comm)
{
  _MPI_Barrier(get_size());
  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */
void *bcast_buffer = nullptr;
int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
  if (get_rank() == root) {
    bcast_buffer = buf;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    bcast_buffer = nullptr;
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    memcpy(buf, bcast_buffer, count * stubtypesize(datatype));
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

/* copy values from data1 to data2 */

int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                  MPI_Comm comm)
{
  MPI_Reduce(sendbuf, recvbuf, count, datatype, op, 0, comm);
  MPI_Bcast(recvbuf, count, datatype, 0, comm);
  
  return MPI_SUCCESS;
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

std::map<int, void*> reduce_buffermap;
std::atomic<int64_t> reduce_counter = 0;
int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm)
{
  // Reset scan counter
  if (get_rank() == 0) {
    reduce_counter = 0;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Write send buffers to map, one thread at a time
  while (reduce_counter < get_rank()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  reduce_buffermap[get_rank()] = sendbuf;
  reduce_counter++;
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (get_rank() == root) {
    memcpy(recvbuf, sendbuf, count * stubtypesize(datatype));

    for (int i = 0; i < get_size(); i++) {
      if (i == root) {
        continue;
      }
      if (datatype == MPI_INT) {
        reduce_op(reinterpret_cast<int*>(recvbuf), reinterpret_cast<int*>(reduce_buffermap[i]), count, op);
      }
      if (datatype == MPI_FLOAT) {
        reduce_op(reinterpret_cast<float*>(recvbuf), reinterpret_cast<float*>(reduce_buffermap[i]), count, op);
      }
      if (datatype == MPI_DOUBLE) {
        reduce_op(reinterpret_cast<double*>(recvbuf), reinterpret_cast<double*>(reduce_buffermap[i]), count, op);
      }
      if (datatype == MPI_CHAR) {
        reduce_op(reinterpret_cast<char*>(recvbuf), reinterpret_cast<char*>(reduce_buffermap[i]), count, op);
      }
      if (datatype == MPI_BYTE) {
        printf("MPI_SUM of MPI_BYTE is not supported");
      }
      if (datatype == MPI_LONG) {
        reduce_op(reinterpret_cast<long*>(recvbuf), reinterpret_cast<long*>(reduce_buffermap[i]), count, op);
      }
      if (datatype == MPI_LONG_LONG) {
        reduce_op(reinterpret_cast<long long*>(recvbuf), reinterpret_cast<long long*>(reduce_buffermap[i]), count, op);
      }
      if (datatype == MPI_DOUBLE_INT) {
        printf("MPI_SUM of MPI_DOUBLE_INT is not supported");
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  return MPI_SUCCESS;
}

/* ---------------------------------------------------------------------- */
std::map<int, void*> scan_buffermap;
std::atomic<int64_t> scan_counter = 0;
int MPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
             MPI_Comm comm)
{
  // Reset scan counter
  if (get_rank() == 0) {
    scan_counter = 0;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Write send buffers to map, one thread at a time
  while (scan_counter < get_rank()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  scan_buffermap[get_rank()] = recvbuf;
  memcpy(recvbuf, sendbuf, count * stubtypesize(datatype));
  
  scan_counter+=1;
  MPI_Barrier(MPI_COMM_WORLD);

  // Reset scan counter
  if (get_rank() == 0) {
    scan_counter = 0;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Calculate sum of previous rank and this, one thread at a time
  while (scan_counter < get_rank()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  if (get_rank() > 0) {
    if (datatype == MPI_INT) {
      reduce_op(reinterpret_cast<int*>(recvbuf), reinterpret_cast<int*>(scan_buffermap[get_rank()-1]), count, op);
    }
    if (datatype == MPI_FLOAT) {
      reduce_op(reinterpret_cast<float*>(recvbuf), reinterpret_cast<float*>(scan_buffermap[get_rank()-1]), count, op);
    }
    if (datatype == MPI_DOUBLE) {
      reduce_op(reinterpret_cast<double*>(recvbuf), reinterpret_cast<double*>(scan_buffermap[get_rank()-1]), count, op);
    }
    if (datatype == MPI_CHAR) {
      reduce_op(reinterpret_cast<char*>(recvbuf), reinterpret_cast<char*>(scan_buffermap[get_rank()-1]), count, op);
    }
    if (datatype == MPI_BYTE) {
      printf("MPI_SUM of MPI_BYTE is not supported");
    }
    if (datatype == MPI_LONG) {
      reduce_op(reinterpret_cast<long*>(recvbuf), reinterpret_cast<long*>(scan_buffermap[get_rank()-1]), count, op);
    }
    if (datatype == MPI_LONG_LONG) {
      reduce_op(reinterpret_cast<long long*>(recvbuf), reinterpret_cast<long long*>(scan_buffermap[get_rank()-1]), count, op);
    }
    if (datatype == MPI_DOUBLE_INT) {
      printf("MPI_SUM of MPI_DOUBLE_INT is not supported");
    }
  }
  scan_counter+=1;
  MPI_Barrier(MPI_COMM_WORLD);
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
