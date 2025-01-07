g++ -O2 -g -pg -o myprog myprog.cpp
./myprog
gprof myprog gmon.out > analysis.txt