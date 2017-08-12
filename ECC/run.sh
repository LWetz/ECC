rm a.out
g++ *.cpp *.h ARFFparser/*.cpp -I/usr/local/cuda-8.0/include -L/usr/local/cuda-8.0/lib64 -lOpenCL -std=c++11
./a.out
