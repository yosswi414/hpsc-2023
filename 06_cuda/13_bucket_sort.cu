#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init_bucket(int* bucket, int range){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= range) return;
    bucket[idx] = 0;
}

__global__ void count(int* key, int* bucket, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;
    atomicAdd(&bucket[key[idx]], 1);
}

int main() {
  const int n = 5000;
  const int range = 50;
//   std::vector<int> key(n);
  int *key;
  cudaMallocManaged(&key, n * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  for(int i = 0; i < 30; ++i)printf("%d ",key[n / 30 * i]);
  printf("\n");

//   std::vector<int> bucket(range); 
  int *bucket;
  cudaMallocManaged(&bucket, range * sizeof(int));
//   for (int i=0; i<range; i++) {
//     bucket[i] = 0;
//   }
  
  int tpb = 1024; // threads_per_block
  
//   init_key<<<(n + tpb - 1) / tpb, tpb>>>(key, n, range);
  init_bucket<<<(n + tpb - 1) / tpb, tpb>>>(bucket, range);
  cudaDeviceSynchronize();

//   for(int i = 0; i < 30; ++i)printf("%d ",key[n / 30 * i]);
//   printf("\n");

//   for (int i=0; i<n; i++) {
//     bucket[key[i]]++;
//   }
  
  count<<<(n + tpb - 1) / tpb, tpb>>>(key, bucket, n);

  cudaDeviceSynchronize();

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

//   for (int i=0; i<n; i++) {
//     printf("%d ",key[i]);
//   }
  for(int i = 0; i < 30; ++i)printf("%d ", key[n / 30 * i]);
  printf("\n");
  
  cudaFree(bucket);
  cudaFree(key);
}
