# 6. Getting the Right Answer
正确答案的获取显然是所有计算的主要目标。在并行系统中，可能会遇到传统串行编程中不常见的困难。这些问题包括线程问题、由于浮点值计算方式导致的意外值，以及 CPU 和 GPU 处理器操作方式差异带来的挑战。本章探讨了可能影响返回数据正确性的问题，并提出了相应的解决方案。

## 6.1. Verification

### 6.1.1. Reference Comparison
对现有程序进行修改时，验证正确性的一个关键方面是建立一种机制，通过这种机制可以将具有代表性的输入的先前已知良好的参考输出与新结果进行比较。每次修改后，使用适用于特定算法的标准确保结果一致。一些算法要求逐位一致的结果，但这并不总是可能的，特别是在涉及浮点运算时；关于数值准确性，请参见《Numerical Accuracy and Precision》。对于其他算法，如果它们在某个小的 epsilon 范围内与参考匹配，则可以认为实现是正确的。

请注意，用于验证数值结果的过程也可以很容易地扩展到验证性能结果。我们希望确保每次所做的更改都是正确的，并且改进了性能（以及改进的程度）。将这些检查作为循环 APOD 过程的一个组成部分，可以帮助确保我们尽快达到预期结果。

### 6.1.2. Unit Testing
一个有用的对比参考方法是将代码结构化，以便在单元级别上轻松验证。例如，我们可以将 CUDA 内核编写为多个简短的 __device__ 函数的集合，而不是一个大型的单片 __global__ 函数；每个设备函数都可以在组合到一起之前独立测试。 

例如，许多内核在进行实际计算的同时还具有复杂的内存访问逻辑。如果我们在引入大规模计算之前单独验证我们的寻址逻辑，这将简化后续的调试工作。（请注意，CUDA 编译器将任何不对全局内存写入的设备代码视为无用代码，会被消除。因此，为了成功应用这一策略，我们必须至少将我们的寻址逻辑结果写入全局内存。）

更进一步，如果大多数函数定义为 __host__ __device__ 而不仅仅是 __device__ 函数，那么这些函数可以在 CPU 和 GPU 上进行测试，从而增加我们对函数正确性的信心，并确保结果之间不会有意外差异。如果存在差异，那么这些差异会早期显现，并且可以在简单函数的上下文中理解。

作为一个有用的副作用，如果我们希望在应用程序中包括 CPU 和 GPU 执行路径，这种策略将允许我们减少代码重复：如果 CUDA 内核的大部分工作是在 __host__ __device__ 函数中完成的，我们可以轻松地从主机代码和设备代码中调用这些函数，而不会产生重复代码。

###  6.2. Debugging
CUDA-GDB 是 GNU 调试器的一个移植版本，可以在 Linux 和 Mac 上运行；详见：https://developer.nvidia.com/cuda-gdb。

NVIDIA Nsight Visual Studio 版适用于 Microsoft Windows 7、Windows HPC Server 2008、Windows 8.1 和 Windows 10，作为 Microsoft Visual Studio 的免费插件提供；详见：https://developer.nvidia.com/nsight-visual-studio-edition。

此外，几个第三方调试器也支持 CUDA 调试；详见：https://developer.nvidia.com/debugging-solutions 了解更多详情。

### 6.3. Numerical Accuracy and Precision

不正确或意外的结果主要源于浮点精度问题，这是由于浮点值的计算和存储方式造成的。以下部分解释了主要的关注点。关于浮点算术的其他特殊性，请参阅《CUDA C++ 编程指南》的功能和技术规格，以及关于浮点精度和性能的白皮书和随附的网络研讨会，详细信息请访问 https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus。

#### 6.3.1. Single vs. Double Precision
计算能力为 1.3 及以上的设备原生支持双精度浮点值（即 64 位宽的值）。由于双精度算术的更高精度和舍入问题，使用双精度算术获得的结果通常会与单精度算术执行的相同操作产生差异。因此，重要的是确保比较相同精度的值，并在一定容差范围内表示结果，而不是期望它们完全一致。

#### 6.3.2. Floating Point Math Is not Associative
每个浮点算术操作都涉及一定量的舍入。因此，算术操作的执行顺序非常重要。如果 A、B 和 C 是浮点值，(A+B)+C 并不能保证等于 A+(B+C)，如同在符号数学中一样。当你并行化计算时，你可能会改变操作顺序，因此并行结果可能不会与顺序结果匹配。这种限制不仅仅特定于 CUDA，而是浮点值并行计算的固有部分。

#### 6.3.3. IEEE 754 Compliance
所有 CUDA 计算设备都遵循 IEEE 754 标准的二进制浮点表示，但有一些小的例外。这些例外在《CUDA C++ 编程指南》的功能和技术规格中详细说明，可能导致与主机系统上计算的 IEEE 754 值不同的结果。

一个关键的差异是融合乘加（FMA）指令，它将乘法和加法操作组合到单一指令执行中。其结果通常会与分别执行这两个操作获得的结果略有不同。

#### 6.3.4. x86 80-bit Computations
x86 处理器在执行浮点计算时可以使用 80 位双扩展精度数学运算。这些计算的结果经常会与在 CUDA 设备上执行的纯 64 位运算结果有所不同。为了让数值更接近，可以将 x86 主机处理器设置为使用常规的双精度或单精度（分别为 64 位和 32 位）。这可以通过 FLDCW x86 汇编指令或等效的操作系统 API 来实现。










