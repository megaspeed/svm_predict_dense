
#include "common.cpp"
#include "svm_data.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
__global__ void dot_prod_dense(float *X, float *Z, int nrows, int ncols)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int shift = gridDim.x*blockDim.x;
	if (i < nrows)
	{
		float buf = 0;

		for (int j = 0; j < ncols; ++j)
		{			
			buf +=X[i*ncols+j]*X[i*ncols+j];
		}
		Z[i] = buf;

	}
	__syncthreads();
}
// C = X * Y[i] : i = 0..nrows-1
__global__ void dot_line(float *X, float *Y, float *Z, int nrows, int ncols)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int shift = gridDim.x*blockDim.x;
	if (i < nrows)
	{
		float buf = 0;

		for (int j = 0; j < ncols; ++j)
		{			
			buf +=X[j]*Y[i*ncols+j];
		}
		Z[i] = buf;

	}
	__syncthreads();
}


__global__ void reduction( float* d_k, float *d_dotSV, float *d_dotTV, float *d_koef, int nSV, int irow, int offset, float gamma, int kernelcode, float *result)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int shift = gridDim.x*blockDim.x;
	const unsigned int blockdim = blockDim.x;
	const unsigned int tid = threadIdx.x;
	__shared__ float reduction [MAXTHREADS];
	if (i < nSV)
	{
		if(kernelcode == 0)	
		{
			float val =  gamma * (2*d_k[irow*nSV+i]-d_dotSV[i]-d_dotTV[irow+offset]);
			reduction[tid] =  d_koef[i]*expf(val);
		}
	}
	__syncthreads();

	if(blockdim>=512)	{if(tid<256){reduction[tid] += reduction[tid + 256];}__syncthreads();}
	if(blockdim>=256)	{if(tid<128){reduction[tid] += reduction[tid + 128];}__syncthreads();}
	if(blockdim>=128)   {if(tid<64)	{reduction[tid] += reduction[tid + 64];}__syncthreads();}
	if(tid<32){	if(blockdim >= 64)	{reduction[tid] += reduction[tid + 32];}
	if(blockdim >= 32)	{reduction[tid] += reduction[tid + 16];}
	if(blockdim >= 16)	{reduction[tid] += reduction[tid + 8];}
	if(blockdim >= 8)	{reduction[tid] += reduction[tid + 4];}
	if(blockdim >= 4)	{reduction[tid] += reduction[tid + 2];}
	if(blockdim >= 2)	{reduction[tid] += reduction[tid + 1];}	}

	if(tid==0){	result[blockIdx.x]=reduction[tid];}
}

void classifier(svm_model *model, svm_test *test, int *h_l_estimated )
{
	int nTV = test->nTV;
	int nSV = model->nSV;
	int nfeatures = model->nfeatures;

	float *d_TV = 0;
	

	float *d_SV = 0;
	cudaMalloc((void**) &d_SV, nSV*nfeatures*sizeof(float));
	cudaMemcpy(d_SV, model->SV_dens, nSV*nfeatures*sizeof(float),cudaMemcpyHostToDevice);

	float *d_l_SV = 0;
	cudaMalloc((void**) &d_l_SV, nSV*sizeof(float));
	cudaMemcpy(d_l_SV, model->l_SV, nSV*sizeof(float),cudaMemcpyHostToDevice);

	float *d_dotTV = 0;
	cudaMalloc((void**) &d_dotTV, nTV*sizeof(float)); 

	float *d_dotSV = 0;
	cudaMalloc((void**) &d_dotSV, nSV*sizeof(float)); 

	// Allocate device memory for F
	


	void* temp;
	size_t pitch;
	cudaMallocPitch(&temp, &pitch, nSV * sizeof(float),1);
	cudaFree(temp);
	unsigned int remainingMemory = 0;
	unsigned int totalMemory = 0;
	cudaMemGetInfo(&remainingMemory, &totalMemory);	
	int cache_size = remainingMemory/pitch; // # of TVs in cache
	if (nTV <= cache_size)
	{
		cache_size = nTV;
	}
	float *d_k = 0;
	cudaMalloc((void**)&d_k, cache_size*nSV * sizeof(float));
	cudaMalloc((void**) &d_TV, cache_size*nfeatures*sizeof(float));
	int nthreads = MAXTHREADS;
	int nblocksSV = min(MAXBLOCKS, (nSV + nthreads - 1)/nthreads);
	int nblocksTV = min(MAXBLOCKS, (nTV + nthreads - 1)/nthreads);
	int nblocks_cache = min(MAXBLOCKS, (cache_size + nthreads - 1)/nthreads);

	float* h_fdata= (float*) malloc(nblocks_cache*sizeof(float));
	float* d_fdata=0;
	cudaMalloc((void**) &d_fdata, nblocks_cache*sizeof(float));
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE);

	/*float* dot= (float*) malloc(nSV*nTV*sizeof(float));
	float* dot1= (float*) malloc(nSV*nTV*sizeof(float));
	float* dot2= (float*) malloc(nSV*sizeof(float));
	for(int i= 0; i<nSV; i++)
		cublasSdot_v2(handle, nfeatures, &d_SV[i*nfeatures], 1, &d_SV[i*nfeatures], 1, &d_dotSV[i]);
	for(int i= 0; i<nTV; i++)
		cublasSdot_v2(handle, nfeatures, &d_TV[i*nfeatures], 1, &d_TV[i*nfeatures], 1, &d_dotTV[i]);	*/
	dot_prod_dense<<<nblocksSV, nthreads>>>(d_SV, d_dotSV, nSV, nfeatures);
	dot_prod_dense<<<nblocksTV, nthreads>>>(d_TV, d_dotTV, nTV, nfeatures);
	int cc=0;
	int offset = 0;
	float alfa = 1.;
	float betta = 0;
	int num_of_parts =  (nTV + cache_size - 1)/cache_size;
	for (int ipart = 0; ipart < num_of_parts; ipart++)
	{
		if ((ipart == (num_of_parts - 1)) && ((nTV - offset) != 0) )
		{
			cache_size = nTV - offset;
		}
			//Allocate Kernel Cache Memory
		cudaMemcpy(&d_TV[offset*nfeatures], test->TV, cache_size*nfeatures*sizeof(float),cudaMemcpyHostToDevice);
		for (int i = 0; i < cache_size; i++)
		{
			dot_line<<<nblocks_cache, nthreads>>>(&d_TV[i*nfeatures], d_SV, &d_k[i*nSV], cache_size, nfeatures);
			//cudaMemcpy(dot, &d_k[i*nSV], nSV*sizeof(float), cudaMemcpyDeviceToHost);
			/*for (int l = 0; l < nSV; l++)
			{
				cublasSdot_v2(handle, nfeatures, &d_TV[i*nfeatures], 1, &d_SV[l*nfeatures], 1, &d_k[i*nSV+l]);
			}*/

			reduction<<<nblocks_cache, nthreads>>>(d_k, d_dotSV, d_dotTV, d_l_SV, nSV, i, offset, model->coef_gamma, model->kernel_type, d_fdata);
			cudaMemcpy(h_fdata, d_fdata, nblocks_cache*sizeof(float), cudaMemcpyDeviceToHost);
			float sum = 0;
			for (int k = 0; k < nblocks_cache; k++)
			{
				sum += h_fdata[k];
			}
			sum += model->b[0];
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
	cublasDestroy(handle);
}
int main(int argc, char **argv)
{
	FILE *input;
	argc = 4;
	argv[1] = "C:\\Data\\b.txt";
	argv[2] = "C:\\Data\\b.model";
	argv[3] = "10";

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

	int* h_estimated_labels = (int*)malloc(model->nSV*sizeof(int));
	
	classifier(model, test, h_estimated_labels);

	int errors=0;

	for (int i=0; i<test->nTV; i++)
	{
		if( test->l_TV[i]!=h_estimated_labels[i])
		{
			errors++;
		}
	}
	printf("# of testing samples %d, # errors %d, Rate %f\n", test->nTV, errors, 100* (float) (test->nTV -errors)/(float)test->nTV);

	return 0;
}
