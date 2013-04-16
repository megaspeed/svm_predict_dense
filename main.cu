
#include "common.cpp"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"

# define cudaCheck\
 {\
 cudaError_t err = cudaGetLastError ();\
 if ( err != cudaSuccess ){\
 printf(" cudaError = '%s' \n in '%s' %d\n", cudaGetErrorString( err ), __FILE__ , __LINE__ );\
 exit(0);}}

__global__ void dot_prod_dense(float *X, float *Z, int nrows, int ncols)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nrows)
	{
		float buf = 0;

		for (int j = 0; j < ncols; ++j)
		{			
			buf +=X[i*ncols+j]*X[i*ncols+j];
		}
		Z[i] = buf;
	}
}
__global__ void dot_line(float *X, float *Y, float *Z, int nrows, int ncols)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nrows)
	{
		float buf = 0;

		for (int j = 0; j < ncols; ++j)
		{			
			buf +=X[j]*Y[i*ncols+j];
		}
		Z[i] = buf;
	}
}
//__global__ void reduction( float* d_k, float *d_dotSV, float *d_dotTV, float *d_koef, int nSV, int irow, int offset, float gamma, int kernelcode, float *result)
//{
//	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//	const unsigned int blockdim = blockDim.x;
//	const unsigned int tid = threadIdx.x;
//	__shared__ float reduction [MAXTHREADS];
//	if (i < nSV)
//	{
//		if(kernelcode == 0)	
//		{
//			float val =  gamma * (2*d_k[irow*nSV+i]-d_dotSV[i]-d_dotTV[irow+offset]);
//			reduction[tid] =  d_koef[i]*expf(val);
//		}
//	}
//	__syncthreads();
//	if (i < nSV)
//	{
//
//		for (int s = blockDim.x/2; s > 32; s >>= 1)
//		{
//			if (tid < 32)	
//				reduction[tid] += reduction[tid + s];
//			__syncthreads();
//		}
//		if(tid<32)
//		{
//			reduction[tid] += reduction[tid + 32];
//			reduction[tid] += reduction[tid + 16];
//			reduction[tid] += reduction[tid + 8];
//			reduction[tid] += reduction[tid + 4];
//			reduction[tid] += reduction[tid + 2];
//			reduction[tid] += reduction[tid + 1];	
//		}
//		if(tid==0){	result[blockIdx.x]=reduction[tid];}
//	}
//}
__global__ void reduction1( float *d_SV, float *d_TV, float *d_koef, int nSV, int ncol, float gamma, int kernelcode, float *result)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int blockSize = blockDim.x;
	const unsigned int tid = threadIdx.x;
	extern __shared__  float reduction [];
	reduction[tid]= 0;
	while (i < nSV)
	{
		if(kernelcode == 0)	
		{
			float val = 0;
			for (int j = 0; j < ncol; j++)
			{
				val += (d_TV[j]-d_SV[i*ncol+j])*(d_TV[j]-d_SV[i*ncol+j]);
			}
			reduction[tid] +=  d_koef[i]*expf(-gamma*val);
		}
		i += blockSize*gridDim.x;
	}
	__syncthreads();
		if(blockSize>=512)	{if(tid<256){reduction[tid] += reduction[tid + 256];}__syncthreads();}
	if(blockSize>=256)	{if(tid<128){reduction[tid] += reduction[tid + 128];}__syncthreads();}
	if(blockSize>=128)  {if(tid<64)	{reduction[tid] += reduction[tid + 64];}__syncthreads();}
	if(tid<32){	if(blockSize >= 64)	{reduction[tid] += reduction[tid + 32];}
				if(blockSize >= 32)	{reduction[tid] += reduction[tid + 16];}
				if(blockSize >= 16)	{reduction[tid] += reduction[tid + 8];}
				if(blockSize >= 8)	{reduction[tid] += reduction[tid + 4];}
				if(blockSize >= 4)	{reduction[tid] += reduction[tid + 2];}
				if(blockSize >= 2)	{reduction[tid] += reduction[tid + 1];}	}
	if(tid==0){	result[blockIdx.x]=reduction[0];}

}

void classifier(svm_model *model, svm_test *test, int *h_l_estimated )
{
	float reductiontime = 0;
	float intervaltime;
	cudaEvent_t start, stop;
	cudaEventCreate ( &start );cudaCheck
	cudaEventCreate ( &stop  );cudaCheck

	int nTV = test->nTV;
	int nSV = model->nSV;
	int nfeatures = model->nfeatures;

	float *d_TV = 0;	
	float *d_SV = 0;
	cudaMalloc((void**) &d_SV, nSV*nfeatures*sizeof(float));cudaCheck
	cudaMemcpy(d_SV, model->SV_dens, nSV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);cudaCheck

		float *d_l_SV = 0;
	cudaMalloc((void**) &d_l_SV, nSV*sizeof(float));cudaCheck
	cudaMemcpy(d_l_SV, model->l_SV, nSV*sizeof(float),cudaMemcpyHostToDevice);cudaCheck

	size_t remainingMemory = 0;
	size_t totalMemory = 0;
	cudaMemGetInfo(&remainingMemory, &totalMemory);	cudaCheck
	int cache_size = remainingMemory/(nSV*sizeof(float)); // # of TVs in cache
	if (nTV <= cache_size){	cache_size = nTV; }

	cudaMalloc((void**) &d_TV, cache_size*nfeatures*sizeof(float));cudaCheck

	int nthreads = MAXTHREADS;
	int nblocks_cache = min(MAXBLOCKS, (cache_size + nthreads - 1)/nthreads);
	int nblocks_SV = min(MAXBLOCKS, (nSV + nthreads - 1)/nthreads);
	dim3 dim_block = dim3(nblocks_cache, 1, 1);
	dim3 dim_thread = dim3(MAXTHREADS, 1, 1);
	// Allocate device memory for F
	float* h_fdata= (float*) malloc(nblocks_SV*sizeof(float));
	float* d_fdata=0;
	cudaMalloc((void**) &d_fdata, nblocks_SV*sizeof(float));cudaCheck
	int offset = 0;
	int num_of_parts =  (nTV + cache_size - 1)/cache_size;

	for (int ipart = 0; ipart < num_of_parts; ipart++)
	{
		if ((ipart == (num_of_parts - 1)) && ((nTV - offset) != 0) )
		{
			cache_size = nTV - offset;
		}
		cudaMemcpy(d_TV, &test->TV[offset*nfeatures], cache_size*nfeatures*sizeof(float),cudaMemcpyHostToDevice);cudaCheck
			for (int i = 0; i < cache_size; i++)
			{				
				reduction1<<<nblocks_SV, MAXTHREADS, MAXTHREADS*sizeof(float)>>>(d_SV, &d_TV[i*nfeatures], d_l_SV, nSV, nfeatures, model->coef_gamma, model->kernel_type, d_fdata);cudaCheck
				cudaMemcpy(h_fdata, d_fdata, nblocks_SV*sizeof(float), cudaMemcpyDeviceToHost); cudaCheck

				float sum = 0;
				for (int k = 0; k < nblocks_SV; k++)
					sum += h_fdata[k];

				sum -= model->b[0];
				if (sum > 0)
				{
					h_l_estimated[i + offset] = model->label_set[0];
				}
				else
				{
					h_l_estimated[i + offset] = model->label_set[1];
				}
			}
			offset += cache_size;
	}
	cudaFree(d_fdata);cudaCheck
	cudaFree(d_l_SV);cudaCheck
	cudaFree(d_SV);cudaCheck
	cudaFree(d_TV);cudaCheck
	cudaDeviceReset();cudaCheck
}
int main(int argc, char **argv)
{
	FILE *input;
	if (argc==1)
	{
	argc = 4;
	//argv[1] = "C:\\Data\\b.txt";
	//argv[2] = "C:\\Data\\b.model";
	//argv[3] = "10";
	//argv[1] = "..\\Data\\a9a.t";
	//argv[2] = "..\\Data\\a9a.model";
	//argv[3] = "123";
	argv[1] = "..\\Data\\cod.t";
	argv[2] = "..\\Data\\cod.model";
	argv[3] = "8";
	//argv[1] = "..\\Data\\cov.t";
	//argv[2] = "..\\Data\\cov.model";
	//argv[3] = "54";
	//argv[1] = "..\\Data\\gisette_scale.t";
	//argv[2] = "..\\Data\\gisette.model";
	//argv[3] = "5000";
	}

	if(argc<4)
		exit_with_help();
	struct svm_model *model = (svm_model*)malloc(sizeof(svm_model));
	struct svm_test *test = (svm_test*)malloc(sizeof(svm_test));
	sscanf(argv[3],"%d",&model->nfeatures);

	if(read_model(argv[2], model, model->nfeatures)==0)
	{
		fprintf(stderr,"can't read model %s\n",argv[2]);
		exit(1);
	}

	if((input = fopen(argv[1],"r")) == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[1]);
		exit(1);
	}
	parse_TV(input,&test->TV,&test->l_TV,&test->nTV,model->nfeatures);
	fclose(input);

	int* h_estimated_labels = (int*)malloc(test->nTV*sizeof(int));

	cuResetTimer();
	classifier(model, test, h_estimated_labels);	
	float time = cuGetTimer();

	int errors=0;
	for (int i=0; i<test->nTV; i++)
	{
		if( test->l_TV[i]!=h_estimated_labels[i])
		{
			errors++;
		}
	}
	printf("# of testing samples %d, # errors %d, Rate %f Time: %f\n", test->nTV, errors, 100* (float) (test->nTV -errors)/(float)test->nTV, time);
	free(h_estimated_labels);
	free(model->b);
	free(model->label_set);
	free(model->l_SV);
	free(model->SV_dens);
	free(model);
	free(test->l_TV);
	free(test->TV);
	free(test);
	return 0;
}
