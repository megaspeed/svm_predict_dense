#ifndef _SVM_DATA_H_
#define _SVM_DATA_H_
#define MAXTHREADS 128
#define MAXSHAREDMEM 49152
#define MAXBLOCKS MAXSHAREDMEM/MAXTHREADS
#define min(a, b)  (((a) < (b)) ? (a) : (b))

struct svm_sample
{
	int nTV;				/*# of test vectors/samples */
	int *l_TV;				/*	TV's labels				*/
	float *TV;				/*	TVs in dense format		*/
};

struct svm_model
{
	int nr_class;		/*	number of classes		*/
	int nSV;			/*	# of SV					*/
	int nfeatures;		/*	# of SV's features		*/
	float *SV_dens;		/*	SVs in dense format		*/
	float *l_SV;		/*	SV's labels				*/
	float *b;			/*	classification parametr	*/	
	int *label_set;		/*  intput lables			*/
	int svm_type;
	int kernel_type;
	float coef_d;
	float coef_gamma;
	float coef_b;
	float C;
	float *params;		/*	params C_i, gamma_i for RBF*/
	int	ntasks;
};

#endif

#ifdef _WIN32

#include <windows.h>

static LARGE_INTEGER t;
static float         f;
static int           freq_init = 0;

void cuResetTimer(void) {
  if (!freq_init) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    f = (float) freq.QuadPart;
    freq_init = 1;
  }
  QueryPerformanceCounter(&t);
}

float cuGetTimer(void) {
  LARGE_INTEGER s;
  float d;
  QueryPerformanceCounter(&s);

  d = ((float)(s.QuadPart - t.QuadPart)) / f;

  return (d*1000.0f);
}

#else

#include <sys/time.h>

static struct timeval t;

/**
 * Resets timer
 */
void cuResetTimer() {
  gettimeofday(&t, NULL);
}


/**
 * Gets time since reset
 */
float cuGetTimer() { // result in miliSec
  static struct timeval s;
  gettimeofday(&s, NULL);

  return (s.tv_sec - t.tv_sec) * 1000.0f + (s.tv_usec - t.tv_usec) / 1000.0f;
}

#endif