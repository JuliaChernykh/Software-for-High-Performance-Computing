#include <iostream>
#include "omp.h"
#include <iomanip>

using namespace std;

int main(){
  double a = 0;
  double b = 1;
  int steps = 1000;
  double step = (b - a) / steps;
  double res = 0;

  for (int i = 0; i < steps; i++)
  {
	  double x = a + (i + 0.5) * step;
	  res += (4.0 / (1.0 + x * x)) * step;
  }
  cout << "1 thread: " << res << endl;

  long double res2 = 0;
  #pragma omp parallel for shared(a,steps,step)
  for (int i = 0; i<steps;i++){
        long double x = a + (i+0.5)*step;
        res2 += f(x) * step;
  }
  cout << "multi-thread without reduction: " << res2 << endl;


  long double res3 = 0;
  #pragma omp parallel for shared(a,steps,step) reduction(+:res3)
  for (int i = 0; i<steps;i++){
        long double x = a + (i+0.5)*step;
        res3 += f(x) * step;
  }
  cout << "multi-thread with reduction: " << res3 << endl;

  return 0;
}
