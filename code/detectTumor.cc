#include<iostream>
#include<cstring>
#include"Bitmap.h"
using namespace std;

string Append(string , string data = "1_Result_");

int main()
{
	string fname;
	cout <<"Give Filename--: ";
	cin >>fname;

	Bitmap mainImg, copyImg;
	int threshold;
	mainImg.open(fname);
	mainImg.gray();
	copyImg = mainImg;
	threshold = mainImg.standardDeviation();
	mainImg.binarizeImage(threshold);
	mainImg.connectedComponent();
	mainImg.multiply(copyImg);
	mainImg.gauSmooth();
	mainImg.histogramEqualisation();
	mainImg.segFCM();
	mainImg.connectedComponent();
	mainImg.multiply(copyImg);
	mainImg.boundary();
	copyImg.clap(mainImg);

	fname = Append(fname);

	copyImg.save(fname, 1);

	return 0;
}


string Append(string fname, string data)
{
	data.append(fname);
	data.append(".bmp");

	return(data);	
}



