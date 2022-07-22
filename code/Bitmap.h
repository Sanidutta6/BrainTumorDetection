/*************************
/ Bitmap Digital Image Handing Library(v2.10)
/ Author:
/ Time: 7/Sep/2019/11:45 AM
/ Purpose: For Detecting Brain Tumor
/ Platform: Ubuntu 19.04 LTS
/*************************/



#include<iostream>
#include<cstdlib>
#include<fstream>
#include<cstring>
#include<cmath>
#include<iomanip>
#include<vector>

using namespace std;

#pragma pack(push)
#pragma pack(2)
struct BMPFileStructure
{
	//BitMapFileHeader--:
	uint16_t type{0x4D42};
	uint32_t size1{0};
	uint16_t reserved1{0};
	uint16_t reserved2{0};
	uint32_t ofBits{0};

	//BitMapInfoHeader--:
	uint32_t size2{0};
	uint32_t width{0};
	uint32_t height{0};
	uint16_t planes{1};
	uint16_t bitCount{0};
	uint32_t compression{0};
	uint32_t sizeImage{0};
	uint32_t xPxlPerMeter{0};
	uint32_t yPxlPerMeter{0};
	uint32_t clrUsed{0};
	uint32_t clrImportant{0};
};
#pragma pack(pop)

struct pixel
{
	unsigned int B, G, R;
};

class Bitmap
{
	BMPFileStructure bmp;
	uint8_t *others = NULL;
	pixel **pixelMatrix;
	int e, rowsize;
	vector<int> stack;
	protected:
		int xGradient(int x, int y);
		int yGradient(int x, int y);
		void check(int );
		void calc(int ,int ,int ,int );
	public:
		Bitmap(void){ };
		Bitmap(int ,int ,int );
		Bitmap(string fname){ open(fname); };
		//IMage information--:
		void open(string );
		void save(string ,int del);
		void details(void);
		int getHeight(void){ return(bmp.height); };
		int getWidth(void){ return(bmp.width); };
		void operator=(Bitmap );
		pixel getPixel(int ,int );
		void setPixel(int ,int ,pixel );
		void savePixels(string ,int ,int ,int ,int );
		//Basic Operation--:
		void gray(void);
		void complement(void);
		void subtruct(Bitmap );
		void multiply(Bitmap );
		int standardDeviation(void);
		void binarizeImage(int );
		void connectedComponent(int component = 1);
		void boundary(void);
		void clap(Bitmap );
		//Filter--:
		void gauSmooth(void);
		void histogram(string );
		void histogramEqualisation(void);
		void detectESobel(int );
		//Sagmentation--:
		void segFCM(void);

};

Bitmap :: Bitmap(int height, int width, int bitCount = 3)
{
	if(height <= 0 || width <= 0)
	{
		cout <<"Height and Width must be positive integer--:\n";
		exit(0);
	}
	//BitmapFileHeader--:
	bmp.type = 0x4d42;
	bmp.size1 = (height*width*bitCount)+54;
	bmp.reserved1 = 0;
	bmp.reserved2 = 0;
	bmp.ofBits = 54;
	//BitmapInfoHeader--:
	bmp.size2 = 55;
	bmp.width = width;
	bmp.height = height;
	bmp.planes = 1;
	e = bmp.bitCount = bitCount;
	bmp.compression = 0;
	bmp.sizeImage = height*width*bitCount;

	rowsize = bmp.width*bitCount;
	pixelMatrix = new pixel *[height];
	for(int i = 0;i < height;i++)
	{
		pixelMatrix[i] = new pixel[width];
		for(int j = 0;j < width;j++)
		{
			pixelMatrix[i*rowsize+j] = 0;
		}
	}
}
//Opening, closing and pixels--:
void Bitmap :: open(string fname)
{
	ifstream fin;
	fin.open(fname, ios::binary);
	fin.read((char *)&bmp, sizeof(bmp));
	if(bmp.type != 0x4d42)
	{
		cout <<"Not a Bitmap file--:\n";
		fin.close();
		exit(1);
	}
	if((bmp.ofBits - 54) != 0)
	{
		others = new uint8_t[bmp.ofBits-54];
		fin.read((char *)others, (bmp.ofBits-54));
	}
	e = (int)bmp.bitCount/8;
	fin.seekg(bmp.ofBits);
	pixelMatrix = new pixel *[bmp.height];
	for(int i = 0;i < bmp.height;i++)
	{
		pixelMatrix[i] = new pixel[bmp.width];
		for(int j = 0;j < bmp.width;j++)
		{
			if(fin.good())
			{
				pixelMatrix[i][j].B = fin.get();
				pixelMatrix[i][j].G = fin.get();
				pixelMatrix[i][j].R = fin.get();
			}
			else
			{
				cout <<"Can't read FILE["<<fname<<"]--:\n";
				exit(1);
			}
		}
	}
	fin.close();
}
void Bitmap :: save(string fname, int del)
{
	uint8_t tmp;
	ofstream fout;
	fout.open(fname, ios::binary);
	fout.write((char *)&bmp, sizeof(bmp));
	if((bmp.ofBits-54) != 0)
	{
		fout.write((char *)&others, (bmp.ofBits-54));
	}
	fout.seekp(bmp.ofBits);
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			tmp = (pixelMatrix[i][j].B > 255)?255:((pixelMatrix[i][j].B < 0)?0:pixelMatrix[i][j].B);
			fout.put(tmp);
			tmp = (pixelMatrix[i][j].G > 255)?255:((pixelMatrix[i][j].G < 0)?0:pixelMatrix[i][j].G);
			fout.put(tmp);
			tmp = (pixelMatrix[i][j].R > 255)?255:((pixelMatrix[i][j].R < 0)?0:pixelMatrix[i][j].R);
			fout.put(tmp);
		}
	}
	fout.close();
	if(del)
	{
		for(int i = 0;i < bmp.height;i++)
		{
			delete(pixelMatrix[i]);
		}
		if(others != NULL)
		{
			delete(others);
		}
	}
}
void Bitmap :: details(void)
{
	cout <<"BMPFIleHeader--:\n";
	cout <<"\tType--:"<<hex<<bmp.type<<endl;
	cout <<"\tSize--:"<<dec<<bmp.size1<<endl;
	cout <<"\tOffset--:"<<dec<<bmp.ofBits<<endl;
	cout <<"BMPInfoHeader--:\n";
	cout <<"\tHeader_Size--:"<<bmp.size2<<endl;
	cout <<"\tWidth--:"<<bmp.width<<endl;
	cout <<"\tHeight--:"<<bmp.height<<endl;
	cout <<"\tPlanes--:"<<bmp.planes<<endl;
	cout <<"\tBit Count--:"<<bmp.bitCount<<endl;
	cout <<"\tCompression--:"<<bmp.compression<<endl;
	cout <<"\tSize of Image--:"<<bmp.sizeImage<<endl;
	cout <<"\tPixels per meter--:"<<bmp.xPxlPerMeter<<endl;
}
void Bitmap :: operator=(Bitmap img)
{
	bmp = img.bmp;
	e = img.e;
	if((img.bmp.ofBits - 54) != 0)
	{
		others = new uint8_t[img.bmp.ofBits-54];
		others = img.others;
	}
	pixelMatrix = new pixel *[bmp.height];
	for(int i = 0;i < bmp.height;i++)
	{
		pixelMatrix[i] = new pixel[bmp.width];
		for(int j = 0;j < bmp.width;j++)
		{
			pixelMatrix[i][j] = img.pixelMatrix[i][j];
		}
	}
}
pixel Bitmap :: getPixel(int h, int w)
{
	if((h > bmp.height || h < bmp.height) || (w > bmp.width || w < bmp.width))
	{
		pixel tmp;
		tmp.R = tmp.G = tmp.B = -1;
		return(tmp);
	}
	return(pixelMatrix[h][w]);
}
void Bitmap :: setPixel(int h, int w, pixel color)
{
	if((h > bmp.height || h < bmp.height) || (w > bmp.width || w < bmp.width))
	{
		cout <<"Check height and width before assigning pixel--:\n";
		return;
	}
	pixelMatrix[h][w] = color;
}
void Bitmap :: savePixels(string fname,int firstRow,int firstCol,int lastRow,int lastCol)
{
	fstream file;
	file.open(fname, ios::out);
	if(!file.good())
	{
		cout <<"ERROR--:\n";
		return;
	}
	file <<"Height--:"<<bmp.height<<endl;
	file <<"Width--:"<<bmp.width<<endl;
	file <<"Bit Count--:"<<bmp.bitCount<<endl;
	file <<"{";
	for(int i = firstRow;i < lastRow;i++)
	{
		file <<"{";
		for(int j = firstCol;j < lastCol;j++)
		{
			file <<pixelMatrix[i][j].R;	//saving red pixel intensity--:
			if(j != bmp.width-1)
			{
				file <<", ";
			}
		}
		file <<"}";
		if(i != bmp.height-1)
		{
			file <<", ";
		}
		file <<"\n";
	}
	file <<"};\n";
	file.close();
}
//Basic Operation--:
void Bitmap :: gray(void)
{
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			int gray = 0.0722*pixelMatrix[i][j].B;
			gray += 0.7152*pixelMatrix[i][j].G;
			gray += 0.2126*pixelMatrix[i][j].R;
			pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = gray;
		}
	}
}
void Bitmap :: complement(void)
{
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			pixelMatrix[i][j].B = 255 - pixelMatrix[i][j].B;
			pixelMatrix[i][j].G = 255 - pixelMatrix[i][j].G;
			pixelMatrix[i][j].R = 255 - pixelMatrix[i][j].R;
		}
	}
}
void Bitmap :: multiply(Bitmap img)
{
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			pixelMatrix[i][j].B *= img.pixelMatrix[i][j].B;
			pixelMatrix[i][j].G *= img.pixelMatrix[i][j].G;
			pixelMatrix[i][j].R *= img.pixelMatrix[i][j].R;
		}
	}
}
void Bitmap :: subtruct(Bitmap img)
{
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			pixelMatrix[i][j].B -= img.pixelMatrix[i][j].B;
			pixelMatrix[i][j].G -= img.pixelMatrix[i][j].G;
			pixelMatrix[i][j].R -= img.pixelMatrix[i][j].R;
		}
	}
}
int Bitmap :: standardDeviation(void)
{
	float sum = 0, mean, sD = 0;
	int tmp, dp = 0;
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			sum += pixelMatrix[i][j].B;
		}
	}
	mean = sum/(bmp.height*bmp.width);
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			tmp = pixelMatrix[i][j].B;
			sD += pow((tmp-mean), 2);
		}
	}
	sum = sqrt(sD/(bmp.height*bmp.width));
	return(sum);
}
void Bitmap :: binarizeImage(int threshold)
{
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			int tmp = pixelMatrix[i][j].R;
			tmp = (tmp > threshold)?255:0;
			pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = tmp;
		}
	}
}
void Bitmap :: connectedComponent(int component)
{
	int *area, side[4], label = 0, tmp, t = 0, max = -1;
	vector<int> equivlncArr;

	//Storing value zero for label 0--:
	equivlncArr.push_back(0);
	//Labeling 'pixelMatrix' and storing labels in 'equivlncArr' vector--:
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			if(pixelMatrix[i][j].R != 0 /*255*/)	//if pixel value having value greater than 0--:
			{
				if(i != 0)		//Upper row--:
				{
					side[0] = (j == 0)?0:pixelMatrix[i-1][j-1].R;	//Upper row leftmost pixel--:
					side[1] = pixelMatrix[i-1][j].R;		//Upper row middle pixel--:
					side[2] = (j == bmp.width)?0:pixelMatrix[i-1][j+1].R;	//Upper row right most pixel--:
				}
				else
				{
					side[0] = side[1] = side[2] = 0;	// When (i-1) < 0 
				}
				side[3] = (j == 0)?0:pixelMatrix[i][j-1].R;	//Current row left most pixel--:
				int min = 9999;
				for(int k = 0;k < 4;k++)
				{
					//Finding minimum label which is also must be non-zero--:
					if(side[k] > 0 && min > side[k])
					{
						min = side[k];
					}
				}
				if(min != 9999)
				{
					//Updating with minimum label for label 'tmp'--:
					for(int k = 0;k < 4;k++)
					{
						tmp = side[k];
						equivlncArr[tmp] = equivlncArr[min];
					}
				}
				else
				{
					//Creating new label--:
					min = ++label;
					equivlncArr.push_back(label);
				}
				pixelMatrix[i][j].R = pixelMatrix[i][j].G = pixelMatrix[i][j].B = min;	//Labling--:
			}
		}
	}
	//Allocating memory to area of size of 'label';
	area = new int[label];
	for(int i = 0;i < label;i++)
	{
		area[i] = 0;
	}
	//Relabeling 'pixelMatrix' and calculating area for each label--:
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			tmp = pixelMatrix[i][j].R;
			if(tmp)
			{
				pixelMatrix[i][j].R = pixelMatrix[i][j].G = pixelMatrix[i][j].B = equivlncArr[tmp];	//Relabling--:
				area[equivlncArr[tmp]]++;	//Calculating area--:
			}
		}
	}
	equivlncArr.clear();	//Deleting equivlncArr--:
	//Finding maximum area and storing index of max area in 'tmp' variable--:
	for(int k = 1;k < label;k++)
	{
		//cout <<k<<"--: "<<area[k]<<endl;
		if(max < area[k])
		{
			max = area[k];	//number of pixel's in max area--:
			t = k;		//Label of max area--:
		}
	}
	cout <<"Max area--: "<<max<<"\tLabel--: "<<t<<endl;
	//Labeling max area (indicated by label 't') to 1 and all others to zero--:
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			tmp = pixelMatrix[i][j].R;
			tmp = (tmp == t)?1:0;	//if pixel label equal to max area's label--:
			pixelMatrix[i][j].R = pixelMatrix[i][j].G = pixelMatrix[i][j].B = tmp;

		}
	}
	delete(area);
}
void Bitmap :: boundary(void)
{
	int threshold = standardDeviation();
	binarizeImage(threshold);
	detectESobel(0);
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			if(pixelMatrix[i][j].R)
			{
				pixelMatrix[i][j].R = 255;
			}
		}
	}
}
void Bitmap :: clap(Bitmap img)
{
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			if(img.pixelMatrix[i][j].R)
			{
				pixelMatrix[i][j].R = img.pixelMatrix[i][j].R;
				pixelMatrix[i][j].G = pixelMatrix[i][j].B = 0;
			}
		}
	}
}
void Bitmap :: gauSmooth(void)
{
	int GConst = 1000;
	//Gaussian Kernel--: Sigma=0.9, Sum=996, Size=7x7
	int filter[7][7] = {{0,0,0,1,0,0,0},
                            {0,1,9,17,9,1,0},
                            {0,9,57,106,57,9,0},
                            {1,17,106,196,106,17,1},
                            {0,9,57,106,57,9,0},
                            {0,1,9,17,9,1,0},
                            {0,0,0,1,0,0,0}};
	for(int i = 3;i < bmp.height-3;i++)
	{
		for(int j = 3;j < bmp.width-3;j++)
		{
			int sum = 0;
			for(int k = -3;k <= 3;k++)
			{
				for(int l = -3;l <= 3;l++)
				{
					int gval = pixelMatrix[i+k][j+l].B;
					int fval = filter[k+3][l+3];
					sum += gval*fval;
				}
			}
			sum /= GConst;
			sum = (sum > 255)?255:((sum < 0)?0:sum);
			pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = sum;
		}
	}
}
void Bitmap :: histogram(string fname)
{
	//Bitmap img(512, 512);
	int frequency[256] = {0}/*, xOrigin = 100, yOrigin = 100*/;
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			int tmp = pixelMatrix[i][j].B;
			frequency[tmp]++;
		}
	}
	cout <<"Histogram--:\n";
	for(int i = 0;i < 256;i++)
	{
		cout <<"Histogram["<<i<<"]--:"<<frequency[i]<<endl;
	}
}
void Bitmap :: histogramEqualisation(void)
{
	float CN[256] = {0}, add = 0;
	int frequency[256] = {0}, total = 0, tmp;
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			tmp = pixelMatrix[i][j].B;
			frequency[tmp]++;
		}
	}
	for(int i = 1;i < 256;i++)
	{
		if(frequency[i])
		{
			total += frequency[i];
		}
	}
	for(int i = 1;i < 256;i++)
	{
		CN[i] = (float)frequency[i]/total;
		add += CN[i];
		CN[i] = add;
	}
	for(int i = 1;i < 256;i++)
	{
		CN[i] *= 255;
		if(CN[i]-ceil(CN[i]) > 0.5)
		{
			CN[i] = ceil(CN[i])+1;
		}
		else
		{
			CN[i] = ceil(CN[i]);
		}
	}
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			tmp = pixelMatrix[i][j].B;
			if(frequency[tmp])
			{
				frequency[tmp]--;
				pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = (int)CN[tmp];
			}
		}
	}
}
void Bitmap :: detectESobel(int threshold = 0)
{
	int gx[bmp.height][bmp.width] = {0}, gy[bmp.height][bmp.width] = {0};
	int max = -20, min = 2000;
	for(int i = 1;i < bmp.height-1;i++)
	{
		for(int j = 1;j < bmp.width-1;j++)
		{
			//Horizontal--:
			gx[i][j] = pixelMatrix[i-1][j-1].B
				  +2*pixelMatrix[i-1][j].B
				  +pixelMatrix[i-1][j+1].B
				  -pixelMatrix[i+1][j-1].B
				  -2*pixelMatrix[i+1][j].B
				  -pixelMatrix[i+1][j+1].B;
			gy[i][j] = pixelMatrix[i-1][j-1].B
				  +2*pixelMatrix[i][j-1].B
				  +pixelMatrix[i+1][j-1].B
				  -pixelMatrix[i-1][j+1].B
				  -2*pixelMatrix[i][j+1].B
				  -pixelMatrix[i+1][j+1].B;
		}
	}
	//Magnitude--:
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			int tmp = sqrt(pow(gx[i][j], 2)+pow(gy[i][j], 2));
			pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = tmp;
			if(tmp > max)
			{
				max = tmp;
			}
			if(min > tmp)
			{
				min = tmp;
			}
		}
	}
	int diff = max-min;
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			int tmp = 255*((pixelMatrix[i][j].B-min)/(diff*1.0));
			//pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = tmp;
			pixelMatrix[i][j].B = pixelMatrix[i][j].G = 0;
			pixelMatrix[i][j].R = tmp;
		}
	}
}
void Bitmap :: segFCM(void)
{
	int dP = 0, *cordinate, cluster, preCluster = 0;
	float **DOM, **preDOM, **distance, *centroid, preCentroid[5] = {0};
	float vIndex, pre_vIndex = 0, tmp, converge = 0;
	//Counting number of data points--:
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			if(pixelMatrix[i][j].B)
			{
				dP++;
			}
		}
	}
	//Assigning memory for arrays--:
	cordinate = new int[dP];
	preDOM = new float *[dP];
	DOM = new float *[dP];
	distance = new float *[dP];
	//assigning data points to cordinate for clustering--:
	for(int i = 0, k = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			if(pixelMatrix[i][j].B)
			{
				cordinate[k++] = pixelMatrix[i][j].B;
			}
		}
	}
	for(cluster = 2;cluster < 6;cluster++)
	{
		cout <<"Cluster--:"<<cluster<<endl;
		float sum, prev;
		int itr = -1;
		//Allocating memory a/c to cluster value to array--:
		centroid = new float[cluster];
		for(int i = 0;i < dP;i++)
		{
			preDOM[i] = new float[cluster];
			DOM[i] = new float[cluster];
			distance[i] = new float[cluster];
		}
		//[STEP-1]: Randomly initialize degreeOfmembership matrix--:
		for(int i = 0;i < dP;i++)
		{
			sum = 0;
			do {
				for(int j = 0;j < cluster;j++)
				{
					DOM[i][j] = 0;
					DOM[i][j] = (rand()%19)-1;
					sum += pow(DOM[i][j], 2);
				}
				sum = sqrt(sum);
			}while(sum == 0);
			tmp = 0;
			for(int j = 0;j < cluster;j++)
			{
				DOM[i][j] /= sum;
				DOM[i][j] *= DOM[i][j];
			}
		}
		//LOOP--:
		do {
			prev = converge;
			//[STEP-2] : Find centroid for each clusters--:
			for(int k = 0;k < cluster;k++)
			{
				sum = 0;
				for(int i = 0;i < dP;i++)
				{
					sum += pow(DOM[i][k], 2);
				}
				centroid[k] = 0;
				for(int i = 0;i < dP;i++)
				{
					centroid[k] += ((cordinate[i]*pow(DOM[i][k], 2))/sum);
				}
				//[STEP-3}: Find distance between cordinate & centroid--:
				for(int i = 0;i < dP;i++)
				{
					distance[i][k] = pow(cordinate[i]-centroid[k], 2);
				}
			}
			//Copying current degreeOfMembership to preDOM--:
			for(int i = 0;i < dP;i++)
			{
				for(int j = 0;j < cluster;j++)
				{
					preDOM[i][j] = DOM[i][j];
				}
			}
			//[STEP-4}: Update the degreeOfMembership matrix--:
			converge = 0;
			for(int i = 0;i < dP;i++)
			{
				sum = 0;
				for(int k = 0;k < cluster;k++)
				{
					tmp = pow(distance[i][k], 2);
					sum += (1/tmp);
				}
				for(int j = 0;j < cluster;j++)
				{
					tmp = pow(distance[i][j], 2);
					DOM[i][j] = (1/tmp)/sum;
					//Convergence equation--:
					converge += pow(DOM[i][j]- preDOM[i][j], 2);
				}
			}
			//[STEP-5]: Checking for convergence condition occurs or not--:
			converge = sqrt(converge);
			itr++;
			//cout <<"\tIteration--:"<<itr<<endl;
		}while(converge > 0.01 && itr < 1000 && fabs(prev-converge) != 0);
		//Calculating Fuzzy Clustering Validity Index--:
		float nk;
		for(int k = 0;k < cluster;k++)
		{
			sum = nk = 0;
			for(int i = 0;i < dP;i++)
			{
				tmp = pow(DOM[i][k], 2);
				sum += (pow(cordinate[i]-centroid[k], 2) * tmp);	//Nominator term of equaltion;
				nk += DOM[i][k];					//fuzzy cardinality;
			}
			tmp = 0;
			for(int j = 0;j < cluster;j++)
			{
				tmp += pow(centroid[j]-centroid[k], 2);			//additive term on denominator;
				tmp *= nk;
			}
			//tmp *= nk;							//denominator term;
			vIndex += sum/tmp;
		}
		if(vIndex > pre_vIndex)
		{
			for(int i = 0;i < cluster;i++)
			{
				preCentroid[i] = centroid[i];
			}
			pre_vIndex = vIndex;
			preCluster = cluster;
		}
		delete(centroid);
		for(int i = 0;i < dP;i++)
		{
			delete(DOM[i]);
			delete(preDOM[i]);
			delete(distance[i]);
		}
		cout <<"Cluster--:"<<cluster;
		cout <<setprecision(9)<<"\tValidity Index--:"<<vIndex<<endl;
	}
	//[STEP-6]: Defuzzification--:
	int max = 0;
	cout <<"\n\nCluster--:"<<preCluster<<"\n\nPreCentroid[0]--:"<<preCentroid[0];
	for(int i = 1;i < preCluster;i++)
	{
		cout <<"\nPreCentroid["<<i<<"]--:"<<preCentroid[i]<<endl;
		if(preCentroid[max] < preCentroid[i])
		{
			max = i;
		}
	}
	//cout <<"Max--:"<<max<<endl;
	for(int i = 0;i < bmp.height;i++)
	{
		for(int j = 0;j < bmp.width;j++)
		{
			if(pixelMatrix[i][j].B)
			{
				float tmp = (pixelMatrix[i][j].B < preCentroid[max])?0:pixelMatrix[i][j].B;
				pixelMatrix[i][j].B = pixelMatrix[i][j].G = pixelMatrix[i][j].R = (int)tmp;
			}
		}
	}
}
