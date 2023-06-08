#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

using namespace std;
using Arr = vector<double>;
using Mat = vector<vector<double>>;

inline double sq(double x){ return x * x; }

int main(){
    int nx, ny, nt, nit;
    double dx, dy, dt, rho, nu;
    nx = 320; // 41
    ny = 320; // 41
    nt = 10;  // 500
    nit = 50;
    dx = 2 / (nx - 1);
    dy = 2 / (ny - 1);
    dt = 1e-3; // 0.01
    rho = 1;
    nu = .02;
    Arr x(nx), y(ny);
    Mat u(ny, Arr(nx, 0)), v, p, b, X, Y, un, vn, pn;
    v = p = b = X = Y = u;
    for(int i = 0; i < nx; ++i) {
        x[i] = 2.0 * i / (nx - 1);
        for(int j = 0; j < ny; ++j) X[j][i] = x[i];
    }
    for(int i = 0; i < ny; ++i) {
        y[i] = 2.0 * i / (ny - 1);
        for(int j = 0; j < nx; ++j) Y[i][j] = y[i];
    }

    for(int n = 0; n < nt; ++n){
        auto tic = chrono::steady_clock::now();
        for(int j = 1; j < ny-1; ++j){
            for(int i = 1; i < nx-1; ++i){
                b[j][i] = rho * (1 / dt *
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    sq((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                     (v[j][i+1] - v[j][i-1]) / (2 * dx)) - sq((v[j+1][i] - v[j-1][i]) / (2 * dy)));
            }
        }

        for(int it = 0; it < nit; ++it){
            pn = p;
            for(int j = 1; j < ny-1; ++j){
                for(int i = 1; i < nx-1; ++i){
                    p[j][i] = (sq(dy) * (pn[j][i+1] + pn[j][i-1]) +
                           sq(dx) * (pn[j+1][i] + pn[j-1][i]) -
                           b[j][i] * sq(dx) * sq(dy))
                          / (2 * (sq(dx) + sq(dy)));
                }
            }
            for(int j = 1; j < ny-1; ++j){
                p[j][nx-1] = p[j][nx-2];
                p[j][0] = p[j][1];
            }
            p[0] = p[1];
            p[ny-1] = Arr(nx, 0);
        }
        un = u;
        vn = v;
        for(int j = 1; j < ny-1; ++j){
            for(int i = 1; i < nx-1; ++i){
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                               - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                               - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                               + nu * dt / sq(dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                               + nu * dt / sq(dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                               - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                               - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                               + nu * dt / sq(dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                               + nu * dt / sq(dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }
        u[0] = v[0] = v.back() = Arr(nx, 0);
        u.back() = Arr(nx, 1);
        for(int j = 1; j < ny-1; ++j){
            u[j][0] = 0;
            u[j][nx-1] = 0;
            v[j][0] = 0;
            v[j][nx-1] = 0;
        }
        auto toc = chrono::steady_clock::now();
        double time = chrono::duration<double>(toc - tic).count();
        printf("step=%d: %lf s\n", n, time);
    }
}
