clang++ -O3 -std=c++14 src/*.cpp -I/usr/include/python2.7/  main.cpp -L/usr/local/lib/ -lOpenCL -pthread -lpython2.7
./a.out
