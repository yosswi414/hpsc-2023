### To compile 11_cavity.cu
```Shellscript
module purge
module load gcc/8.3.0 cuda/11.0.3
nvcc 11_cavity.cu
```

`test.cu`: almost same code, which won't run on GPU
