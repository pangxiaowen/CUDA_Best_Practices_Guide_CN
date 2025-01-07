# 4. Parallelizing Your Application
在确定热点并进行基本练习以设定目标和期望后，开发人员需要将代码并行化。根据原始代码，这可能就像调用现有的 GPU 优化库（如 cuBLAS、cuFFT 或 Thrust）一样简单，或者仅需添加一些预处理指令作为并行化编译器的提示。

另一方面，一些应用程序的设计需要一定程度的重构以揭示其固有的并行性。即使是 CPU 架构也需要揭示这种并行性以提高或仅仅维持顺序应用程序的性能，CUDA 系列并行编程语言（CUDA C++、CUDA Fortran 等）旨在尽可能简单地表达这种并行性，同时在设计为最大并行吞吐量的支持 CUDA 的 GPU 上运行。