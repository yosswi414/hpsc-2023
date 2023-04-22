#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 2e7;
  int range = 1e5-1;
  std::vector<int> key(n);
// ↓ このディレクティブをおくと計算時間がシングル << マルチになる なぜ？
// rand() が各スレッドで共通の資源である可能性？
// #pragma omp parallel for shared(key)
#pragma omp single
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    // printf("%d ",key[i]);
  }
  if(n > 20){
    for(int i = 0; i < 5; ++i) printf("%d ", key[i]);
    printf("... ");
    for(int i = n-5; i < n; ++i) printf("%d ", key[i]);
    printf("\n");
  }
  else{
    for(int i = 0; i < n; ++i) printf("%d ", key[i]);
    printf("\n");
  }

  std::vector<int> bucket(range,0); 
#pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;
  std::vector<int> offset(range,0);

  // for (int i=1; i<range; i++) 
  //   offset[i] = offset[i-1] + bucket[i-1];
  
  std::vector<int> cpy(range, 0);
#pragma omp parallel for
  for(int i = 1; i < range; ++i) offset[i] = bucket[i - 1];
  // printf("init ofs: ");
  // for(int i = 0; i < range; ++i) printf("%d ", offset[i]);
  // printf("\n");
#pragma omp parallel
  for(int j = 1; j < range; j <<= 1){
#pragma omp for
    for(int i = 0; i < range; ++i) cpy[i] = offset[i];
#pragma omp for
    for(int i = j; i < range; ++i) offset[i] += cpy[i - j];
  }

  // printf("offset: ");
  // for(int i = 0; i < range; ++i)printf("%d ", offset[i]);
  // printf("\n");
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    // int j = offset[i];
    // for (; bucket[i]>0; bucket[i]--) {
    //   key[j++] = i;
    // }
#pragma omp parallel for
    for(int k = 0; k < bucket[i]; ++k) key[offset[i] + k] = i;
    // while(bucket[i]-- > 0) key[j++] = i;
    // printf("i: %d, j: %d\n", i, j);
  }

//   for (int i=0; i<n; i++) {
//     printf("%d ",key[i]);
//   }
  if(n > 20){
    for(int i = 0; i < 5; ++i) printf("%d ", key[i]);
    printf("... ");
    for(int i = n-5; i < n; ++i) printf("%d ", key[i]);
    printf("\n");
  }
  else{
    for(int i = 0; i < n; ++i)printf("%d ", key[i]);
    printf("\n");
  }
}
