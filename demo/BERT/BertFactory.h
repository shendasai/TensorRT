#include "NvInfer.h"
#if 0
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>

#include <sys/stat.h>
#endif
#include <sstream>

#include <vector>

using namespace std;
using namespace nvinfer1;

namespace bert{

#define create_bert(type)  new Bert##type()


class Bert
{
public:
     Bert(){};
	 Bert(int numHeads, int Bmax, int S, bool runInFp16);
	 virtual ~Bert(); 
	 virtual void init(string weightsPath){};
	 virtual void forward(Weights& inputIds, Weights& segmentIds, Weights& inputMasks, Dims& inputDims, std::vector<float>& output){};

     void setParam(int numHeads, int Bmax, int S, bool runInFp16);
	 int getNumHeads();
	 int getBMax();
	 int getS();
	 bool getRunInFp16();
	 int getDeviceId();
	 string getOutputName();
	 void setOutputName(string outputName);
	 void setDeviceId(int deviceId);
private:
	//string dataDirs_;
	int numHeads_;
	int Bmax_;  //batchsize
	int S_;     //sentence len
	bool runInFp16_;
	int deviceId_;
	//string saveEngine_; 
	//const HostTensorMap inCfg_;
	
	string outputName_;
};



Bert* createBert(string type);



}
	

