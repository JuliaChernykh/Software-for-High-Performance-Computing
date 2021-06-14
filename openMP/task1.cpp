#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// parallel algorithm
int task2_parallel(int** matrix, int m, int n)
{
	int max = 0;
	double t1 = omp_get_wtime();
	for (int i = 0; i < m; ++i)
	{
		int curmin = matrix[i][0];

        #pragma omp parallel for reduction(min: curmin)
		for (int j = 1; j < n; ++j)
		{
		    if (matrix[i][j] < curmin) 
		{
                curmin = matrix[i][j];
            }
		}
		
        if (curmin > max) {
            max = curmin;
        }
	}
	double t2 = omp_get_wtime();
	printf("time (omp): %lf\n", t2 - t1);
	return max;
}

// m*n matrix generation
int** generate_matrix(int m, int n)
{
	int** res = new int*[m];
	srand(time(NULL));
	for (int i = 0; i < m; ++i)
	{
		res[i] = new int[n];
		for (int j = 0; j < n; ++j)
		{
			res[i][j] = rand() % 100;
		}
	}
	return res;
}

int main()
{
	int m = 30, n = 30;
	int** matrix = generate_matrix(m, n);

	printf("max: %d\n", task2_parallel(matrix, m, n));
	
	for (int i = 0; i < m; ++i)
	{
		delete[] matrix[i];
	}
	delete[] matrix;
	return 0;
}
