CUDA_ARCH ?= sm_60

test: test.cu labels.o timer.o centroids.h kmeans.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -o test test.cu timer.o labels.o -lcublas

test_li: test_li.cu labels.o timer.o centroids.h kmeans.h helpers.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -o test_li test_li.cu timer.o labels.o -lcublas -I/usr/include/python2.7 -lpython2.7

test_simple: k-means-simple-thrust.cu helpers.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -o test_simple k-means-simple-thrust.cu -I/usr/include/python2.7 -lpython2.7

test_cpp: k-means-cpp.cpp helpers.h
	g++ -std=c++11 -O3 -o test_cpp k-means-cpp.cpp -I/usr/include/python2.7 -lpython2.7

labels.o: labels.cu labels.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -c -o labels.o labels.cu

timer.o: timer.cu timer.h
	nvcc -arch=$(CUDA_ARCH) -Xptxas -v -c -o timer.o timer.cu