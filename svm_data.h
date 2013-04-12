#ifndef _SVM_DATA_H_
#define _SVM_DATA_H_
#define MAXTHREADS 128
#define MAXBLOCKS 9196
#define KMEM 1

#define min(a, b)  (((a) < (b)) ? (a) : (b))



struct svm_test
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
};

#endif