CXX = g++
CXXFLAGS = -std=c++11 -g -O0 -c -Wall -Wextra
LDFLAGS = -std=c++11 

all : neuro

neuro : neuro.o
	$(CXX) $(LDFLAGS) neuro.o -o output
neuro.o : neuro.cpp
	$(CXX) $(CXXFLAGS) neuro.cpp

clean :
	rm rf *o neuro
