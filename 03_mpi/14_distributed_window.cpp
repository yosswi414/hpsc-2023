#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char** argv) {
  const int N = 20;
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Body ibody[N/size], jbody[N/size];
  srand48(rank);
  for(int i=0; i<N/size; i++) {
    ibody[i].x = jbody[i].x = drand48();
    ibody[i].y = jbody[i].y = drand48();
    ibody[i].m = jbody[i].m = drand48();
    ibody[i].fx = jbody[i].fx = ibody[i].fy = jbody[i].fy = 0;
  }
  int recv_from = (rank + 1) % size;
  int send_to = (rank - 1 + size) % size;
  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);

  MPI_Win win;

  // int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
  // RMA 操作のためのウィンドウオブジェクト生成
  // base: メモリ領域の先頭アドレス
  // size: メモリ領域のサイズ
  // disp_unit: Put/Get で指定されるデータ単位
  // info: 最適化のための情報
  // comm: MPI communicator
  // win: ウィンドウオブジェクト
  MPI_Win_create(jbody, N/size * sizeof(Body), sizeof(Body), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  MPI_Win_fence(0, win);
  //  int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
  //           int target_rank, MPI_Aint target_disp, int target_count,
  //           MPI_Datatype target_datatype, MPI_Win win);
  // ターゲットプロセスへのデータ書き込み
  // origin_addr: 自プロセスのデータ元の先頭アドレス
  // origin_count: データ個数
  // origin_datatype: データ型
  // target_rank: ターゲットプロセスの rank
  // target_disp: ウィンドウ先頭からのオフセット
  // target_count: データ個数
  // target_datatype: データ型
  // win: ウィンドウオブジェクト

  // 受信元確認用ウィンドウ
  int recv[2];
  MPI_Win win_test;
  MPI_Win_create(recv, 2 * sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_test);
  MPI_Win_fence(0, win_test);

  for(int irank=0; irank<size; irank++) {
    // MPI_Send(jbody, N/size, MPI_BODY, send_to, 0, MPI_COMM_WORLD);
    // MPI_Recv(jbody, N/size, MPI_BODY, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Put(jbody, N / size, MPI_BODY,
            send_to, 0, N / size, MPI_BODY, win);
    MPI_Win_fence(0, win);

    // range_begin(受信元確認)
    recv[0] = rank;
    recv[1] = irank;
    MPI_Put(recv, 2, MPI_INT, send_to, 0, 2, MPI_INT, win_test);
    MPI_Win_fence(0, win_test);
    if(recv[0] != recv_from || recv[1] != irank)
      fprintf(stderr, "rank: %d, recv_from/recv[0]: %d/%d, irank/recv[1]: %d/%d\n", rank, recv_from, recv[0], irank, recv[1]);
    // printf("rank: %d, send_to: %d, recv_from: %d, irank: %d, recv[0] = %d\n", rank, send_to, recv_from, irank, recv[0]);
    // range_end(受信元確認)

    for (int i = 0; i < N / size; i++) {
      for(int j=0; j<N/size; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r = std::sqrt(rx * rx + ry * ry);
        if (r > 1e-15) {
          ibody[i].fx -= rx * jbody[j].m / (r * r * r);
          ibody[i].fy -= ry * jbody[j].m / (r * r * r);
        }
      }
    }
  }
  for(int irank=0; irank<size; irank++) {
    // MPI_Barrier(MPI_COMM_WORLD);
    if(irank==rank) {
      for(int i=0; i<N/size; i++) {
        printf("%d %g %g\n",i+rank*N/size,ibody[i].fx,ibody[i].fy);
      }
    }
  }
  // ウィンドウオブジェクト解放
  MPI_Win_free(&win);
  MPI_Win_free(&win_test);
  MPI_Finalize();
}
