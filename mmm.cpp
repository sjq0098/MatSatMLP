#include <iostream>
#include <vector>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>
#include <fstream>   
#include <iomanip> 
using namespace std;

// 获取高精度计时器
LARGE_INTEGER GetPerformanceCounter() {
	LARGE_INTEGER counter;
	QueryPerformanceCounter(&counter);
	return counter;
}

double GetElapsedTime(const LARGE_INTEGER& start,
	const LARGE_INTEGER& end,
	const LARGE_INTEGER& freq) {
		return (end.QuadPart - start.QuadPart) * 1e6 / freq.QuadPart; // microseconds
	}

// 朴素矩阵乘法 C = A x B
void NaiveMM(const vector<vector<double>>& A,
	const vector<vector<double>>& B,
	vector<vector<double>>& C) {
		int N = A.size();
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				double sum = 0;
				for (int k = 0; k < N; ++k) {
					sum += A[i][k] * B[k][j];
				}
				C[i][j] = sum;
			}
		}
	}

// Cache-optimized (i-k-j ordering)
void CacheOptimizedMM(const vector<vector<double>>& A,
	const vector<vector<double>>& B,
	vector<vector<double>>& C) {
		int N = A.size();
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				C[i][j] = 0;
		
		for (int i = 0; i < N; ++i) {
			for (int k = 0; k < N; ++k) {
				double r = A[i][k];
				for (int j = 0; j < N; ++j) {
					C[i][j] += r * B[k][j];
				}
			}
		}
	}

// Unroll k-loop by 4
void UnrollMM(const vector<vector<double>>& A,
	const vector<vector<double>>& B,
	vector<vector<double>>& C) {
		int N = A.size();
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				C[i][j] = 0;
			}
			for (int k = 0; k + 3 < N; k += 4) {
				for (int j = 0; j < N; ++j) {
					C[i][j] += A[i][k] * B[k][j]
					+ A[i][k+1] * B[k+1][j]
					+ A[i][k+2] * B[k+2][j]
					+ A[i][k+3] * B[k+3][j];
				}
			}
			for (int k = (N/4)*4; k < N; ++k) {
				for (int j = 0; j < N; ++j) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
	}

// SIMD using AVX (assume N is multiple of 4)
void SimdMM(const vector<vector<double>>& A,
	const vector<vector<double>>& B,
	vector<vector<double>>& C) {
		int N = A.size();
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				C[i][j] = 0;
			}
			for (int k = 0; k < N; ++k) {
				__m256d a_vec = _mm256_set1_pd(A[i][k]);
				int j = 0;
				for (; j + 3 < N; j += 4) {
					__m256d b_vec = _mm256_loadu_pd(&B[k][j]);
					__m256d c_vec = _mm256_loadu_pd(&C[i][j]);
					c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
					_mm256_storeu_pd(&C[i][j], c_vec);
				}
				for (; j < N; ++j) {
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
	}

// OpenMP parallel version with cache optimization
void OpenMPLoopMM(const vector<vector<double>>& A,
	const vector<vector<double>>& B,
	vector<vector<double>>& C) {
		int N = A.size();
#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				C[i][j] = 0;
			}
		}
#pragma omp parallel for collapse(2)
		for (int i = 0; i < N; ++i) {
			for (int k = 0; k < N; ++k) {
				double r = A[i][k];
				for (int j = 0; j < N; ++j) {
					C[i][j] += r * B[k][j];
				}
			}
		}
	}

// Helper to generate random matrix
void generateMatrix(vector<vector<double>>& M) {
	int N = M.size();
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			M[i][j] = rand() % 100;
		}
	}
}

int main() {
	srand((unsigned)time(NULL));
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	
	vector<int> sizes = {128, 256, 512,1024,2048, 4096};
	 ofstream output("result.txt");
    if (!output) {
        cerr << "无法打开输出文件" << endl;
        return 1;
    }
	for (int N : sizes) {
		vector<vector<double>> A(N, vector<double>(N));
		vector<vector<double>> B(N, vector<double>(N));
		vector<vector<double>> C(N, vector<double>(N));
		generateMatrix(A);
		generateMatrix(B);
		
		// Define lambdas
		struct { const char* name; void(*func)(const vector<vector<double>>&,const vector<vector<double>>&,vector<vector<double>>&);} methods[] = {
			{"Naive", NaiveMM},
			{"CacheOpt", CacheOptimizedMM},
			{"Unroll", UnrollMM},
			{"SIMD", SimdMM},
			{"OpenMP", OpenMPLoopMM}
		};
		
		output << "Matrix Size: " << N << "x" << N << endl;
		for (auto& m : methods) {
			LARGE_INTEGER start = GetPerformanceCounter();
			m.func(A, B, C);
			LARGE_INTEGER end = GetPerformanceCounter();
			double t = GetElapsedTime(start, end, freq);
			output << m.name << " Time: " << fixed << setprecision(2) << t << " us" << endl;
		}
		output << "-------------------------------" << endl;
	}
	
	return 0;
}

