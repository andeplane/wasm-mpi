#ifndef MPI_STUBS
#define MPI_STUBS
#include <map>
#include <barrier>
#include <functional>
#include <thread>
#include <stdlib.h>

#ifndef __cplusplus
#error "MPI STUBS must be compiled with a C++ compiler"
#endif

#define MPI_COMM_WORLD 0

#define MPI_SUCCESS 0
#define MPI_ERR_OTHER 1
#define MPI_ERR_ARG -1

#define MPI_INT 1
#define MPI_FLOAT 2
#define MPI_DOUBLE 3
#define MPI_CHAR 4
#define MPI_BYTE 5
#define MPI_LONG 6
#define MPI_LONG_LONG 7
#define MPI_DOUBLE_INT 8

#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3
#define MPI_MAXLOC 4
#define MPI_MINLOC 5
#define MPI_LOR 6

#define MPI_UNDEFINED -1
#define MPI_COMM_NULL -1
#define MPI_GROUP_EMPTY -1
#define MPI_GROUP_NULL -1

#define MPI_ANY_SOURCE -1
#define MPI_STATUS_IGNORE NULL

#define MPI_Comm int
#define MPI_Datatype int
#define MPI_Op int
#define MPI_Fint int
#define MPI_Group int
#define MPI_Offset long

#define MPI_IN_PLACE NULL

#define MPI_MAX_PROCESSOR_NAME 128
#define MPI_MAX_LIBRARY_VERSION_STRING 128
enum MPI_Request_type {
  MPI_Irecv_t = 0,
  MPI_Send_t = 1,
  MPI_Recv_t = 2
};

struct MPI_RequestKey {
  MPI_Datatype datatype;
  MPI_Request_type type;
  int count;
  int source;
  int dest;
  int tag;
  bool const operator==(const MPI_RequestKey &o) {
    return datatype == o.datatype
       &&  type == o.type
       &&  count == o.count
       &&  source == o.source
       &&  dest == o.dest
       &&  tag == o.tag;
  }

  bool const operator<(const MPI_RequestKey &o) const {
      return datatype < o.datatype
          || (datatype == o.datatype && type < o.type)
          || (datatype == o.datatype && type == o.type && source < o.source)
          || (datatype == o.datatype && type == o.type && source == o.source && dest < o.dest)
          || (datatype == o.datatype && type == o.type && source == o.source && dest == o.dest && tag < o.tag);
  }
};

struct MPI_Request {
  MPI_Datatype datatype;
  MPI_Request_type type;
  int count;
  int source;
  int dest;
  int tag;
  MPI_Comm comm;
  const void *sendbuf;
  void *recvbuf;
};

struct MPI_STATE {
  int num_threads;
  int max_threads;
  std::barrier<> barrier_2=std::barrier(2);
  std::barrier<> barrier_3=std::barrier(3);
  std::barrier<> barrier_4=std::barrier(4);
  std::barrier<> barrier_5=std::barrier(5);
  std::barrier<> barrier_6=std::barrier(6);
  std::barrier<> barrier_7=std::barrier(7);
  std::barrier<> barrier_8=std::barrier(8);
  std::map<std::pair<int, int>, std::shared_ptr<std::barrier<>>> send_barriers;
  std::mutex requestMutex;
  std::map<MPI_RequestKey, MPI_Request> requests;
  MPI_STATE(int num_threads);

  void setRequest(MPI_Request_type type, int source, int dest, MPI_Datatype datatype, int count, int tag, MPI_Request request);
  std::optional<MPI_Request> getRequest(MPI_Request_type type, int source, int dest, MPI_Datatype datatype, int count, int tag);
  void removeRequest(MPI_Request request);
};

void MPI_Reset();
void _MPI_Barrier(int num_threads);
typedef void MPI_User_function(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype);

/* MPI data structs */

struct _MPI_Status {
  int MPI_SOURCE;
};

typedef struct _MPI_Status MPI_Status;

/* Function prototypes for MPI stubs */
void MPI_Register_Thread(int rank, int num_threads);
int MPI_Init(int *argc, char ***argv);
int MPI_Initialized(int *flag);
int MPI_Finalized(int *flag);
int MPI_Get_library_version(char *version, int *resultlen);
int MPI_Get_processor_name(char *name, int *resultlen);
int MPI_Get_version(int *major, int *minor);

int MPI_Comm_rank(MPI_Comm comm, int *me);
int MPI_Comm_size(MPI_Comm comm, int *nprocs);
int MPI_Abort(MPI_Comm comm, int errorcode);
int MPI_Finalize();
double MPI_Wtime();

int MPI_Type_size(int, int *);
int MPI_Request_free(MPI_Request *request);

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
              MPI_Request *request);
int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
             MPI_Status *status);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm,
              MPI_Request *request);
int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Waitall(int n, MPI_Request *request, MPI_Status *status);
int MPI_Waitany(int count, MPI_Request *request, int *index, MPI_Status *status);
int MPI_Sendrecv(const void *sbuf, int scount, MPI_Datatype sdatatype, int dest, int stag,
                 void *rbuf, int rcount, MPI_Datatype rdatatype, int source, int rtag,
                 MPI_Comm comm, MPI_Status *status);
int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *comm_out);
int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *comm_out);
int MPI_Comm_free(MPI_Comm *comm);
MPI_Fint MPI_Comm_c2f(MPI_Comm comm);
MPI_Comm MPI_Comm_f2c(MPI_Fint comm);
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
int MPI_Group_free(MPI_Group *group);

int MPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, int *periods, int reorder,
                    MPI_Comm *comm_cart);
int MPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords);
int MPI_Cart_shift(MPI_Comm comm, int direction, int displ, int *source, int *dest);
int MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int MPI_Type_commit(MPI_Datatype *datatype);
int MPI_Type_free(MPI_Datatype *datatype);

int MPI_Op_create(MPI_User_function *function, int commute, MPI_Op *op);
int MPI_Op_free(MPI_Op *op);

int MPI_Barrier(MPI_Comm comm);
int MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                  MPI_Comm comm);
int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm);
int MPI_Scan(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
             MPI_Comm comm);
int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Allgatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
                   int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts, MPI_Datatype datatype,
                       MPI_Op op, MPI_Comm comm);
int MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
               MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Gatherv(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int *recvcounts,
                int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Scatterv(void *sendbuf, int *sendcounts, int *displs, MPI_Datatype sendtype, void *recvbuf,
                 int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                 MPI_Datatype recvtype, MPI_Comm comm);
int MPI_Alltoallv(void *sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype,
                  void *recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype,
                  MPI_Comm comm);
/* ---------------------------------------------------------------------- */

#endif
