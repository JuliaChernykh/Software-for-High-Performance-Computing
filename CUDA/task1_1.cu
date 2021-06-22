// Задание 1.1. Сложение двух векторов
// Написать программу, которая преобразует последовательный код в код на CUDA C с параметрами ядра 
// <<<1, 1 >>>. Затем изменить параметры ядра <<< N, 1 >>> (blockIdx.x) и <<< 1, N >>> (threadIdx.x)

#include "stdio.h"
#include "assert.h"
#include "math.h"
#include <iostream>
using namespace std;

#define N 5

__global__ void add(int *a, int *b, int *c, int  count) {
	int idx = threadIdx.x;

    if (idx < count)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main( void ) {
	const int maxerror = 1e-6;
        int ha[N], hb[N], hc[N];
	int *da, *db, *dc;

    for (int i = 0; i < N; i++) {
        ha[i] = -i;
        hb[i] = i * i;
    }
        for (int i = 0; i < N; i++) {
       		cout << ha[i] << endl;
        }
	for (int i = 0; i < N; i++) {
		cout << hb[i] << endl;
	}
	// выделяем память на GPU
	cudaMalloc((void**)&da, sizeof(int) * N);
        cudaMalloc((void**)&db, sizeof(int) * N);
        cudaMalloc((void**)&dc, sizeof(int) * N);

	// копируем 
	cudaMemcpy(da, ha, sizeof(int) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, sizeof(int) * N, cudaMemcpyHostToDevice);

	// вызов в GPU
	add<<<1, N>>>(da, db, dc, N);

	// копируем с устройства на хост
	cudaMemcpy(hc, dc, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
    {
        cout << hc[i] << endl;
    }

	for (int i = 0; i < N; i++)
    {
        assert(abs(hc[i] - ha[i] - hb[i]) < maxerror);
    }

	printf("<1, N> case passed\n");

	// копируем 
	cudaMemcpy(da, ha, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(int) * N, cudaMemcpyHostToDevice);

	// вызов в GPU
	add<<<N, 1>>>(da, db, dc, N);

	// копируем с устройства на хост
	cudaMemcpy(hc, dc, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
    {
        cout << hc[i] << endl;
    }

	for (int i = 0; i < N; i++)
    {
        assert(abs(hc[i] - ha[i] - hb[i]) < maxerror);
    }

	printf("<N, 1> case passed\n");

	cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}
