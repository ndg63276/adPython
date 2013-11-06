#include "BufferBitmapSource.h"
#include <iostream>

namespace qrviddec {

BufferBitmapSource::BufferBitmapSource(int inWidth, int inHeight, char * inBuffer) 
: LuminanceSource::LuminanceSource(inWidth, inHeight)
{
	width = inWidth; 
	height = inHeight; 
	buffer = ArrayRef<char>(inBuffer, inWidth * inHeight); 
}

BufferBitmapSource::~BufferBitmapSource()
{
}

int BufferBitmapSource::getWidth() const
{
	return width; 
}

int BufferBitmapSource::getHeight() const
{
	return height; 
}

ArrayRef<char> BufferBitmapSource::getRow(int y, ArrayRef<char> row) const
{
	if (y < 0 || y >= height) 
	{
		fprintf(stderr, "ERROR, attempted to read row %d of a %d height image.\n", y, height); 
		return NULL; 
	}
	// WARNING: NO ERROR CHECKING! You will want to add some in your code. 
	for (int x = 0; x < width; x ++)
	{
		row[x] = buffer[y*width+x]; 
	}
	return row; 
}

ArrayRef<char> BufferBitmapSource::getMatrix() const
{
	return buffer; 
}

}
