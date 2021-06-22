#include "stdio.h"
#include "assert.h"
#include "math.h"
#include <iostream>
using namespace std;

#define N 100000
#define BLOCK_SIZE 1024
#define MAX_ERR 1e-6

__global__ void add(int *a, int *b, int *c, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int *ha, *hb, *hc;
    int *da, *db, *dc;

    ha = (int *)malloc(sizeof(int) * N);
    hb = (int *)malloc(sizeof(int) * N);
    hc = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++)
    {
        ha[i] = -i;
        hb[i] = i * i;
    }

    cudaMalloc((void **)&da, sizeof(int) * N);
    cudaMalloc((void **)&db, sizeof(int) * N);
    cudaMalloc((void **)&dc, sizeof(int) * N);

    cudaMemcpy(da, ha, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(int) * N, cudaMemcpyHostToDevice);

    add<<<(N + BLOCK_SIZE) / BLOCK_SIZE, BLOCK_SIZE>>>(da, db, dc, N);

    cudaMemcpy(hc, dc, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        assert(abs(hc[i] - ha[i] - hb[i]) < MAX_ERR);
    }

    cout << "passed" << endl;

    free(ha);
    free(hb);
    free(hc);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}
