
#include "common.cpp"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"

# define cudaCheck\
 {\
 cudaError_t err = cudaGetLastError ();\
 if ( err != cudaSuccess ){\
 printf(" cudaError = '%s' \n in '%s' %d\n", cudaGetErrorString( err ), __FILE__ , __LINE__ );\
 exit(0);}}

__global__ void reduction( float *d_SV, float *d_TV, float *d_koef, int nSV, int ncol, float gamma, int kernelcode, float *result)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int blockSize = blockDim.x;
	const unsigned int tid = threadIdx.x;
	extern __shared__  float reduct[];
	reduct[tid]= 0;
	while (i < nSV)
	{
		if(kernelcode == 0)	
		{
			float val = 0;
			for (int j = 0; j < ncol; j++)
			{
				val += (d_TV[j]-d_SV[i*ncol+j])*(d_TV[j]-d_SV[i*ncol+j]);
			}
			reduct[tid] += d_koef[i]*expf(-gamma*val);
		}
		i += blockSize*gridDim.x;
	}
	__syncthreads();
		if(blockSize>=512)	{if(tid<256){reduct[tid] += reduct[tid + 256];}__syncthreads();}
	if(blockSize>=256)	{if(tid<128){reduct[tid] += reduct[tid + 128];}__syncthreads();}
	if(blockSize>=128)  {if(tid<64)	{reduct[tid] += reduct[tid + 64];}__syncthreads();}
	if(tid<32){	if(blockSize >= 64)	{reduct[tid] += reduct[tid + 32];}
				if(blockSize >= 32)	{reduct[tid] += reduct[tid + 16];}
				if(blockSize >= 16)	{reduct[tid] += reduct[tid + 8];}
				if(blockSize >= 8)	{reduct[tid] += reduct[tid + 4];}
				if(blockSize >= 4)	{reduct[tid] += reduct[tid + 2];}
				if(blockSize >= 2)	{reduct[tid] += reduct[tid + 1];}	}
	if(tid==0){	result[blockIdx.x]=reduct[0];}

}

void classifier(svm_model *model, svm_sample *test, float *rate)
{
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

	int nblocks_SV = min(MAXBLOCKS, (nSV + MAXTHREADS - 1)/MAXTHREADS);
	// Allocate device memory for F
	float* h_fdata= (float*) malloc(nblocks_SV*sizeof(float));
	float* d_fdata=0;
	cudaMalloc((void**) &d_fdata, nblocks_SV*sizeof(float));cudaCheck
	int offset = 0;
	int num_of_parts =  (nTV + cache_size - 1)/cache_size;
	int* h_l_estimated = (int*)malloc(nTV*sizeof(int));
	for (int ipart = 0; ipart < num_of_parts; ipart++)
	{
		if ((ipart == (num_of_parts - 1)) && ((nTV - offset) != 0) )
		{
			cache_size = nTV - offset;
		}
		cudaMemcpy(d_TV, &test->TV[offset*nfeatures], cache_size*nfeatures*sizeof(float),cudaMemcpyHostToDevice);cudaCheck
			for (int i = 0; i < cache_size; i++)
			{				
				reduction<<<nblocks_SV, MAXTHREADS, MAXTHREADS*sizeof(float)>>>(d_SV, &d_TV[i*nfeatures], d_l_SV, nSV, nfeatures, model->coef_gamma, model->kernel_type, d_fdata);cudaCheck
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

	int errors=0;
	for (int i=0; i<nTV; i++)
	{
		if( test->l_TV[i]!=h_l_estimated[i])
		{
			errors++;
		}
	}
	*rate = (float)(nTV - errors)/nTV;
	
	free(h_l_estimated);
	free(h_fdata);
}
int main(int argc, char **argv)
{
	FILE *input;
	if (argc==1)
	{
	argc = 4;
	argv[1] = "C:\\Data\\b.txt";
	argv[2] = "C:\\Data\\b.model";
	argv[3] = "10";
	//argv[1] = "C:\\Data\\a9a.t";
	//argv[2] = "C:\\\Data\\a9a.model";
	//argv[3] = "123";
	//argv[1] = "C:\\Data\\cod-rna.t";
	//argv[2] = "C:\\Data\\cod.model";
	//argv[3] = "8";
	}

	if(argc<4)
		exit_with_help();
	struct svm_model *model = (svm_model*)malloc(sizeof(svm_model));
	struct svm_sample *test = (svm_sample*)malloc(sizeof(svm_sample));
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

	float rate;
	cuResetTimer();
	classifier(model, test, &rate);
	printf("# of testing samples %d, # errors %d, Rate %f\n", test->nTV, test->nTV - (int)(rate*test->nTV), 100*rate);
	printf("Time: %f\n", cuGetTimer());
	
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
