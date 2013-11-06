#include <zxing/LuminanceSource.h>

#include <stdio.h>
#include <stdlib.h>
using namespace zxing; 
namespace qrviddec {

class BufferBitmapSource : public LuminanceSource {
private:
  int width, height; 
  ArrayRef<char> buffer; 

public:
  BufferBitmapSource(int inWidth, int inHeight, char * inBuffer); 
  ~BufferBitmapSource(); 

  int getWidth() const; 
  int getHeight() const; 
  virtual ArrayRef<char> getRow(int y, ArrayRef<char> row) const; 
  virtual ArrayRef<char> getMatrix() const; 
}; 
}
