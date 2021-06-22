#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void add1D(int* A, int* B, int* C, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = iy * nx + ix;

    C[idx] = A[idx] + B[idx];
}

__global__ void add2D(int* A, int* B, int* C, int nx, int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

__global__ void add1D1D(int* A, int* B, int* C, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ix < nx) {
        for (int iy = 0; iy < ny; ++iy) {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}


__global__ void add2D1D(int* A, int* B, int* C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;

    unsigned int idx = iy*nx + ix;

    if (ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

void sumh (int *A, int *B, int *C, const int nx, const int ny)
{
    int *ia = A;
    int *ib = B;
    int *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void checkres(int *host, int *gpu, const int N)
{
    const double e = 1.0E-6;

    for (int i = 0; i < N; i++)
    {
        if (abs(host[i] - gpu[i]) > e)
        {
            printf("host ", host[i], " gpu ", gpu[i]);
            printf("Test failed\n\n");
            break;
        }
    }
}

void getrandommatrix(int* m, int n)
{
    for (int i=0; i < n; ++i)
        m[i] = rand()% 10;
}

int main( void ) {
    double time1, time2, time3, time4;

    // size of matrix
    unsigned int nx = 1 << 10; // столбцы
    unsigned int ny = 1 << 10; // строки
    int size = nx * ny;

    int* hA = (int*)malloc(size * sizeof(int));
    int* hB = (int*)malloc(size * sizeof(int));
    int* hC = (int*)malloc(size * sizeof(int));
    int* cpuC = (int*)malloc(size * sizeof(int));

    getrandommatrix(hA, size);
    getrandommatrix(hB, size);
    sumh(hA, hB, cpuC, nx, ny);

    int* dA;
    int* dB;
    cudaMalloc((void**)&dA, size * sizeof(int));
    cudaMalloc((void**)&dB, size * sizeof(int));

    cudaMemcpy(dA, hA, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size * sizeof(int), cudaMemcpyHostToDevice);

    printf("Started succesfylly\n");

    // 1D
    int* dC1D;
    cudaMalloc((void**)&dC1D, size * sizeof(int));

    cudaDeviceSynchronize();
    time1 = cpuSecond();
    add1D<<< ny, nx >>>(dA, dB, dC1D, nx, ny);
    cudaDeviceSynchronize();
    time1 = cpuSecond() - time1;
    printf("1D <<<", nx, " ", ny, ">>> elapsed ", time1, " ms\n");

    cudaMemcpy(hC, dC1D, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkres(cpuC, hC, size);
    cudaFree(dC1D);

    // 2D
    int* dC2D;
    cudaMalloc((void**)&dC2D, size * sizeof(int));

    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    cudaDeviceSynchronize();
    time2 = cpuSecond();
    add2D<<<grid, block>>>(dA, dB, dC2D, nx, ny);
    cudaDeviceSynchronize();
    time2 = cpuSecond() - time2;
    printf("2D <<<", grid.x, grid.y, ", ", block.x, block.y, ">>> elapsed ", time2, " ms\n");

    cudaMemcpy(hC, dC2D, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkres(cpuC, hC, size);
    cudaFree(dC2D);

    // 1D-сетка, 1D-блоки
    int* dC1D1D;
    cudaMalloc((void**)&dC1D1D, size * sizeof(int));

    block = dim3{128,1};
    grid = dim3{(nx+block.x-1)/block.x,1};

    cudaDeviceSynchronize();
    time3 = cpuSecond();
    add1D1D <<<grid, block>>> (dA, dB, dC1D1D, nx, ny);
    cudaDeviceSynchronize();
    time3 = cpuSecond() - time3;
    printf("1D1D <<<", grid.x, grid.y, ", ", block.x, block.y, ">>> elapsed ", time3, " ms\n");
    cudaMemcpy(hC, dC1D1D, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkres(cpuC, hC, size);
    cudaFree(dC1D1D);


    // 2D-сетка, 1D-блоки
    int* dC2D1D;
    cudaMalloc((void**)&dC2D1D, size * sizeof(int));

    block = dim3{256};
    grid = dim3{(nx + block.x - 1) / block.x,ny};

    cudaDeviceSynchronize();
    time4 = cpuSecond();
    add2D1D<<<grid, block>>> (dA, dB, dC2D1D, nx, ny);
    cudaDeviceSynchronize();
    time4 = cpuSecond() - time4;
    printf("2D1D <<<", grid.x, grid.y, ", ", block.x, block.y, ">>> elapsed ", time3, " ms\n");
    cudaMemcpy(hC, dC2D1D, size * sizeof(int), cudaMemcpyDeviceToHost);
    checkres(cpuC, hC, size);
    cudaFree(dC2D1D);


    cudaFree(dA);
    cudaFree(dB);

    free(hA);
    free(hB);
    free(hC);

    return 0;
}
