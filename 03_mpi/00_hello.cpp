#include <cstdio>
#include "mpi.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv); // MPI communicator の初期化 (NULL を渡してもよい)
  int size, rank;
  // MPI_COMM_WORLD: MPI communicator (ハンドル) (mpi.h で定義されている)
  MPI_Comm_size(MPI_COMM_WORLD, &size); // プロセス数 size を取得
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // プロセス番号 rank を取得
  printf("rank: %d/%d\n",rank,size);
  MPI_Finalize(); // MPI communicator を終了
}
