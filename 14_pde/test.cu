#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

using namespace std;

double *x, *y;
double *u, *v, *p, *b, *X, *Y, *un, *vn, *pn;

const int nx = 320; // 41
const int ny = 320; // 41
const int nt = 10; // 500
const int nit = 50;
const double dx = 2 / (nx - 1);
const double dy = 2 / (ny - 1);
const double dt = 1e-3; // 0.01
const double rho = 1;
const double nu = .02;

inline double sq(double x){ return x * x; }
inline double calc_b(int i, int j, const double* u, const double* v, const double* b, double dx, double dy, double dt, double rho){
    return rho * (1 / dt *
                    ((u[(j) * nx + i+1] - u[(j) * nx + i-1]) / (2 * dx) + (v[(j+1) * nx + i] - v[(j-1) * nx + i]) / (2 * dy)) -
                    sq((u[(j) * nx + i+1] - u[(j) * nx + i-1]) / (2 * dx)) - 2 * ((u[(j+1) * nx + i] - u[(j-1) * nx + i]) / (2 * dy) *
                     (v[(j) * nx + i+1] - v[(j) * nx + i-1]) / (2 * dx)) - sq((v[(j+1) * nx + i] - v[(j-1) * nx + i]) / (2 * dy)));
}
inline double calc_p(int i, int j, const double* pn, const double* b, double dx, double dy){
    return (sq(dy) * (pn[(j) * nx + i+1] + pn[(j) * nx + i-1]) +
                           sq(dx) * (pn[(j+1) * nx + i] + pn[(j-1) * nx + i]) -
                           b[(j) * nx + i] * sq(dx) * sq(dy))
                          / (2 * (sq(dx) + sq(dy)));
}
inline double calc_u(int i, int j, const double* un, const double* p, double dx, double dy, double dt, double rho, double nu){
    return un[(j) * nx + i] - un[(j) * nx + i] * dt / dx * (un[(j) * nx + i] - un[(j) * nx + i - 1])
                               - un[(j) * nx + i] * dt / dy * (un[(j) * nx + i] - un[(j - 1) * nx + i])
                               - dt / (2 * rho * dx) * (p[(j) * nx + i+1] - p[(j) * nx + i-1])
                               + nu * dt / sq(dx) * (un[(j) * nx + i+1] - 2 * un[(j) * nx + i] + un[(j) * nx + i-1])
                               + nu * dt / sq(dy) * (un[(j+1) * nx + i] - 2 * un[(j) * nx + i] + un[(j-1) * nx + i]);
}
inline double calc_v(int i, int j, const double* vn, const double* p, double dx, double dy, double dt, double rho, double nu){
    return vn[(j) * nx + i] - vn[(j) * nx + i] * dt / dx * (vn[(j) * nx + i] - vn[(j) * nx + i - 1])
                               - vn[(j) * nx + i] * dt / dy * (vn[(j) * nx + i] - vn[(j - 1) * nx + i])
                               - dt / (2 * rho * dx) * (p[(j+1) * nx + i] - p[(j-1) * nx + i])
                               + nu * dt / sq(dx) * (vn[(j) * nx + i+1] - 2 * vn[(j) * nx + i] + vn[(j) * nx + i-1])
                               + nu * dt / sq(dy) * (vn[(j+1) * nx + i] - 2 * vn[(j) * nx + i] + vn[(j-1) * nx + i]);
}

int main(){
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

    for(int n = 0; n < nt; ++n){
        auto tic = chrono::steady_clock::now();
        for(int j = 1; j < ny-1; ++j){
            for(int i = 1; i < nx-1; ++i){
                b[(j) * nx + i] = calc_b(i, j, u, v, b, dx, dy, dt, rho);
            }
        }

        for(int it = 0; it < nit; ++it){
            pn = p;
            for(int j = 1; j < ny-1; ++j){
                for(int i = 1; i < nx-1; ++i){
                    p[(j) * nx + i] = calc_p(i, j, pn, b, dx, dy);
                }
            }
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
        
        for(int j = 1; j < ny-1; ++j){
            for(int i = 1; i < nx-1; ++i){
                u[(j) * nx + i] = calc_u(i, j, un, p, dx, dy, dt, rho, nu);
                v[(j) * nx + i] = calc_v(i, j, vn, p, dx, dy, dt, rho, nu);
            }
        }
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
}
