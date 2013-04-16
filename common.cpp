#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm_data.h"

static char* readline(FILE *input, char* line, int *max_line_len)
{
	int len;
	if(fgets(line,*max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		*max_line_len *= 2;
		line = (char *) realloc(line,*max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,*max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
void free_model(struct svm_model *model)
{
	free(model->SV_dens);
	free(model->l_SV);
	free(model->b);
	free(model);	
}
void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}
/**
* Parses data from file in the libsvm format
* @param inputfilename pointer to the char array that contains the file name
* @param h_xdata host pointer to the array that will store the training set
* @param h_ldata host pointer to the array that will store the labels of the training set
* @param nsamples number of samples in the training set
* @param nfeatures number of features per sample in the training set
* @param nclasses number of classes
*/
int parse_SV(FILE* inputFilePointer, float** h_xdata, float** h_ldata, int nsamples, int nfeatures)
{
	char* stringBuffer = (char*)malloc(65536);
	static char* line;

	*h_xdata = (float*) calloc( nsamples*nfeatures,sizeof(float));
	*h_ldata = (float*) calloc( nsamples,sizeof(float));

	for(int i = 0; i < nsamples; i++)
	{
		char c;
		int pos=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);

			if((c== ' ') || (c == '\n'))
			{
				if(pos==0)
				{
					//Label found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					(*h_ldata)[i]=value;
					pos++;
				}
				else
				{
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					(*h_xdata)[i*nfeatures + (pos-1)]= value;
				}
				bufferPointer = stringBuffer;
			}
			else if(c== ':')
			{
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos= value;
				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');
	}
	free(stringBuffer);
	free(line);
	fclose(inputFilePointer);
	return 1;
}
int parse_TV(FILE* inputFilePointer, float** h_xdata, int** h_ldata, int *nsamples, int nfeatures)
{
	char* stringBuffer = (char*)malloc(65536);
	static int max_line_len = 1024;
	static char* line;
	*nsamples = 0;
	line = (char*)malloc(max_line_len * sizeof(char));
	while(readline(inputFilePointer, line, &max_line_len)!=NULL)
	{
		char *p = strtok(line," \t"); // label
		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
		}
		++(*nsamples);
	}
	rewind(inputFilePointer);
	//*nsamples = NTV;
	*h_xdata = (float*) calloc( *nsamples*nfeatures,sizeof(float));
	*h_ldata = (int*) calloc( *nsamples,sizeof(int));

	for(int i = 0; i < *nsamples; i++)
	{
		char c;
		int pos=0;
		char* bufferPointer = stringBuffer;

		do
		{
			c = fgetc(inputFilePointer);

			if((c== ' ') || (c == '\n'))
			{
				if(pos==0)
				{
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%d", &value);
					(*h_ldata)[i]=value;
					pos++;
				}
				else
				{
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					(*h_xdata)[i*nfeatures + (pos-1)]= value;
				}
				bufferPointer = stringBuffer;
			}
			else if(c== ':')
			{
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos= value;
				bufferPointer = stringBuffer;
			}
			else
			{
				*(bufferPointer) = c;
				bufferPointer++;
			}

		}
		while (c != '\n');
	}
	free(stringBuffer);
	free(line);
	fclose(inputFilePointer);
	return 1;
}
int read_model(const char* model_file_name, svm_model *model, int nfeatures)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return 0;
	const char *svm_type_table[] = { "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",0 };
	const char *kernel_type_table[] = { "rbf","linear","polynomial","sigmoid","precomputed",0 };
	// read parameters
	model->nSV = NULL;
	model->SV_dens = NULL;
	model->l_SV = NULL;
	model->label_set = NULL;
	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0; svm_type_table[i]!=0;i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					model->svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				free_model(model);
				return 0;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					model->kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				free_model(model);
				return 0;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%f",&model->coef_d);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%f",&model->coef_gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%f",&model->coef_b);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->nSV);
		else if(strcmp(cmd,"rho")==0)
		{
			int n = model->nr_class * (model->nr_class-1)/2;
			model->b = (float*)malloc(n*sizeof(float));
			for(int i=0;i<n;i++)
				fscanf(fp,"%f",&model->b[i]);
		}
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label_set = (int*)malloc(n*sizeof(int));
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label_set[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			int temp;
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&temp);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			free_model(model);
			return 0;
		}
	}
	// read sv_coef and SV
	parse_SV(fp,&model->SV_dens,&model->l_SV,model->nSV,nfeatures);
	return 1;
}
void exit_with_help()
{
	printf("Usage: svm-predict test_file model_file #_of_features\n");
	exit(1);
}
