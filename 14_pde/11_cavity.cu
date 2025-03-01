#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

using namespace std;

const int nx = 5120; // 41
const int ny = 5120; // 41
const int nt = 10; // 500
const int nit = 50;
const double dx = 2.0 / (nx - 1);
const double dy = 2.0 / (ny - 1);
const double dt = 1e-3; // 0.01
const double rho = 1;
const double nu = .02;

#define sq(x) ((x) * (x))

__global__ void calc_b_gpu(const double* u, const double* v, double* b){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // k == j * nx + i
    int j = k / nx;
    int i = k % nx;
    if(j <= 0 || ny-1 <= j) return;
    if(i <= 0 || ny-1 <= i) return;

    b[j * nx + i] = rho * (1 / dt *
                    ((u[(j) * nx + i+1] - u[(j) * nx + i-1]) / (2 * dx) + (v[(j+1) * nx + i] - v[(j-1) * nx + i]) / (2 * dy)) -
                    sq((u[(j) * nx + i+1] - u[(j) * nx + i-1]) / (2 * dx)) - 2 * ((u[(j+1) * nx + i] - u[(j-1) * nx + i]) / (2 * dy) *
                     (v[(j) * nx + i+1] - v[(j) * nx + i-1]) / (2 * dx)) - sq((v[(j+1) * nx + i] - v[(j-1) * nx + i]) / (2 * dy)));
}

__global__ void calc_p_gpu(const double* pn, const double* b, double* p){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // k == j * nx + i
    int j = k / nx;
    int i = k % nx;
    if(j <= 0 || ny-1 <= j) return;
    if(i <= 0 || ny-1 <= i) return;

    p[(j) * nx + i] = (sq(dy) * (pn[(j) * nx + i+1] + pn[(j) * nx + i-1]) +
                           sq(dx) * (pn[(j+1) * nx + i] + pn[(j-1) * nx + i]) -
                           b[(j) * nx + i] * sq(dx) * sq(dy))
                          / (2 * (sq(dx) + sq(dy)));
}
__global__ void calc_u_gpu(const double* un, const double* p, double* u){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // k == j * nx + i
    int j = k / nx;
    int i = k % nx;
    if(j <= 0 || ny-1 <= j) return;
    if(i <= 0 || ny-1 <= i) return;

    u[(j) * nx + i] = un[(j) * nx + i] - un[(j) * nx + i] * dt / dx * (un[(j) * nx + i] - un[(j) * nx + i - 1])
                               - un[(j) * nx + i] * dt / dy * (un[(j) * nx + i] - un[(j - 1) * nx + i])
                               - dt / (2 * rho * dx) * (p[(j) * nx + i+1] - p[(j) * nx + i-1])
                               + nu * dt / sq(dx) * (un[(j) * nx + i+1] - 2 * un[(j) * nx + i] + un[(j) * nx + i-1])
                               + nu * dt / sq(dy) * (un[(j+1) * nx + i] - 2 * un[(j) * nx + i] + un[(j-1) * nx + i]);
}
__global__ void calc_v_gpu(const double* vn, const double* p, double* v){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // k == j * nx + i
    int j = k / nx;
    int i = k % nx;
    if(j <= 0 || ny-1 <= j) return;
    if(i <= 0 || ny-1 <= i) return;

    v[(j) * nx + i] = vn[(j) * nx + i] - vn[(j) * nx + i] * dt / dx * (vn[(j) * nx + i] - vn[(j) * nx + i - 1])
                               - vn[(j) * nx + i] * dt / dy * (vn[(j) * nx + i] - vn[(j - 1) * nx + i])
                               - dt / (2 * rho * dx) * (p[(j+1) * nx + i] - p[(j-1) * nx + i])
                               + nu * dt / sq(dx) * (vn[(j) * nx + i+1] - 2 * vn[(j) * nx + i] + vn[(j) * nx + i-1])
                               + nu * dt / sq(dy) * (vn[(j+1) * nx + i] - 2 * vn[(j) * nx + i] + vn[(j-1) * nx + i]);
}


constexpr int ceil(const int a, const int b){ return (a + b - 1) / b; }

int main(){
    double *x, *y;
    double *u, *v, *p, *b, *X, *Y, *un, *vn, *pn;
    cudaMallocManaged(&x, nx * sizeof(double));
    cudaMallocManaged(&y, ny * sizeof(double));
    cudaMallocManaged(&u, nx * ny * sizeof(double));
    cudaMallocManaged(&v, nx * ny * sizeof(double));
    cudaMallocManaged(&p, nx * ny * sizeof(double));
    cudaMallocManaged(&b, nx * ny * sizeof(double));
    cudaMallocManaged(&X, nx * ny * sizeof(double));
    cudaMallocManaged(&Y, nx * ny * sizeof(double));
    cudaMallocManaged(&un, nx * ny * sizeof(double));
    cudaMallocManaged(&vn, nx * ny * sizeof(double));
    cudaMallocManaged(&pn, nx * ny * sizeof(double));
    for(int i = 0; i < nx; ++i) {
        x[i] = 2.0 * i / (nx - 1);
        for(int j = 0; j < ny; ++j) X[(j) * nx + i] = x[i];
    }
    for(int i = 0; i < ny; ++i) {
        y[i] = 2.0 * i / (ny - 1);
        for(int j = 0; j < nx; ++j) Y[(i) * nx + j] = y[i];
    }

    
    const int Thd = 1024;
    const int Blc = ceil(Thd, nx * ny);
    // Blc * Thd > nx * ny

    for(int n = 0; n < nt; ++n){
        auto tic = chrono::steady_clock::now();
        // for(int j = 1; j < ny-1; ++j){
        //     for(int i = 1; i < nx-1; ++i){
        //         b[(j) * nx + i] = calc_b(i, j, u, v, b, dx, dy, dt, rho);
        //     }
        // }
        calc_b_gpu<<<Blc, Thd>>>(u,v,b);
        cudaDeviceSynchronize();
        for(int it = 0; it < nit; ++it){
            pn = p;
            // for(int j = 1; j < ny-1; ++j){
            //     for(int i = 1; i < nx-1; ++i){
            //         p[(j) * nx + i] = calc_p(i, j, pn, b, dx, dy);
            //     }
            // }
            calc_p_gpu<<<Blc, Thd>>>(pn, b, p);
            cudaDeviceSynchronize();

            for(int j = 1; j < ny-1; ++j){
                p[(j) * nx + nx-1] = p[(j) * nx + nx-2];
                p[(j) * nx + 0] = p[(j) * nx + 1];
            }
            for(int i = 1; i < nx-1; ++i){
                p[(0) * nx + i] = p[(1) * nx + i];
                p[(ny-1) * nx + i] = 0;
            }
        }
        for(int j = 1; j < ny-1; ++j){
            for(int i = 1; i < nx-1; ++i){
                un[(j) * nx + i] = u[(j) * nx + i];
                vn[(j) * nx + i] = v[(j) * nx + i];
            }
        }
        
        // for(int j = 1; j < ny-1; ++j){
        //     for(int i = 1; i < nx-1; ++i){
        //         u[(j) * nx + i] = calc_u(i, j, un, p, dx, dy, dt, rho, nu);
        //         v[(j) * nx + i] = calc_v(i, j, vn, p, dx, dy, dt, rho, nu);
        //     }
        // }
        calc_u_gpu<<<Blc, Thd>>>(un, p, u);
        cudaDeviceSynchronize();
        calc_v_gpu<<<Blc, Thd>>>(un, p, u);
        cudaDeviceSynchronize();

        for(int i = 1; i < nx-1; ++i){
            u[(0) * nx + i] = 0;
            v[(0) * nx + i] = 0;
            v[(ny-1) * nx + i] = 0;
            u[(ny-1) * nx + i] = 1;
        }
        for(int j = 1; j < ny-1; ++j){
            u[(j) * nx + 0] = 0;
            u[(j) * nx + nx-1] = 0;
            v[(j) * nx + 0] = 0;
            v[(j) * nx + nx-1] = 0;
        }
        auto toc = chrono::steady_clock::now();
        double time = chrono::duration<double>(toc - tic).count();
        printf("step=%d: %lf s\n", n, time);
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(X);
    cudaFree(Y);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(pn);
}
