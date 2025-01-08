# 8. Performance Metrics
优化 CUDA 代码时，准确衡量性能和了解带宽在性能测量中的作用非常重要。本章讨论了如何使用 CPU 计时器和 CUDA 事件正确测量性能，并探讨了带宽如何影响性能指标以及如何缓解其带来的某些挑战。

## 8.1. Timing
CUDA 调用和内核执行可以使用 CPU 计时器或 GPU 计时器进行计时。这一节探讨了这两种方法的功能、优点和缺点。

### 8.1.1. Using CPU Timers
任何 CPU 计时器都可以用来测量 CUDA 调用或内核执行的时间。虽然各种 CPU 计时方法的细节不在本文档的范围之内，但开发人员应该始终了解其计时方法提供的分辨率。

在使用 CPU 计时器测量 CUDA 调用或内核执行时间时，必须记住，许多 CUDA API 函数是异步的，也就是说，它们在完成工作之前会将控制权返回给调用的 CPU 线程。所有内核启动都是异步的，带有 Async 后缀的内存拷贝函数也是如此。因此，为了准确测量特定调用或一系列 CUDA 调用的耗时，有必要通过在启动和停止 CPU 计时器之前调用 cudaDeviceSynchronize() 来将 CPU 线程与 GPU 同步。cudaDeviceSynchronize() 会阻塞调用的 CPU 线程，直到该线程之前发出的所有 CUDA 调用完成为止。

虽然也可以将 CPU 线程与 GPU 上的特定流或事件同步，但这些同步功能不适用于在默认流之外的流中进行计时代码。cudaStreamSynchronize() 会阻塞 CPU 线程，直到给定流中之前发出的所有 CUDA 调用完成为止。cudaEventSynchronize() 会阻塞，直到 GPU 记录了特定流中的给定事件。因为驱动程序可能会交错执行来自其他非默认流的 CUDA 调用，因此这些调用可能会包含在计时中。

由于默认流（stream 0）在设备上的工作表现出序列化行为（默认流中的操作只有在任何流中所有前面的调用完成后才能开始；在它完成之前，任何流中的后续操作都不能开始），因此可以可靠地在默认流中使用这些函数进行计时。

请注意，像本节中提到的 CPU 到 GPU 的同步点会导致 GPU 处理流水线停滞，因此应谨慎使用它们以尽量减少对性能的影响。

### 8.1.2. Using CUDA GPU Timers
CUDA 事件 API 提供了一些调用，能够创建和销毁事件、记录事件（包括时间戳）以及将时间戳差异转换为以毫秒为单位的浮点值。如何使用 CUDA 事件对代码进行计时说明了它们的用法。

* How to time code using CUDA events
```
cudaEvent_t start, stop;
float time;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord( start, 0 );
kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y,
                           NUM_REPS);
cudaEventRecord( stop, 0 );
cudaEventSynchronize( stop );

cudaEventElapsedTime( &time, start, stop );
cudaEventDestroy( start );
cudaEventDestroy( stop );
```

在这里，cudaEventRecord() 用于将开始和停止事件放入默认流（stream 0）中。设备在流中到达该事件时将为该事件记录一个时间戳。cudaEventElapsedTime() 函数返回从记录开始事件到停止事件之间的经过时间。该值以毫秒表示，分辨率约为半微秒。与此列表中的其他调用一样，它们的具体操作、参数和返回值在 CUDA 工具包参考手册中进行了描述。请注意，计时是在 GPU 时钟上测量的，因此计时分辨率是操作系统独立的。

## 8.2. Bandwidth
带宽，即数据传输的速率，是影响性能的最重要因素之一。几乎所有代码的更改都应考虑它们如何影响带宽。如本指南的内存优化部分所述，带宽可以受到数据存储的内存选择、数据的布局方式以及访问顺序等因素的显著影响。

为了准确衡量性能，计算理论带宽和有效带宽是很有用的。当有效带宽远低于理论带宽时，设计或实现细节可能会降低带宽，因此后续优化工作的主要目标应该是提高带宽。

```
高优先级：在衡量性能和优化效益时，将计算的有效带宽作为指标。
```

### 8.2.1. Theoretical Bandwidth Calculation

理论带宽可以使用产品文献中提供的硬件规格来计算。例如，NVIDIA Tesla V100 使用 HBM2（双倍数据速率）RAM，内存时钟频率为 877 MHz，内存接口宽度为 4096 位。

使用这些数据项，NVIDIA Tesla V100 的峰值理论内存带宽为 898 GB/s：

$$ \text{理论带宽} = \frac{\text{内存时钟频率} \times \text{接口宽度} \times \text{DDR因子}}{\text{换算单位}} $$

其中：

- 内存时钟频率：877 MHz
- 内存接口宽度：4096 位
- DDR因子：2（因为 HBM2 是双倍数据速率）
- 换算单位：1GB = \(10^9\) 字节

代入上述公式进行计算：

$$ 898 GB/s = \frac{877 \times 10^6 \times 4096 \times 2}{10^9} $$

在此计算中，内存时钟频率转换为 Hz，乘以接口宽度（除以 8，将位转换为字节），并因双倍数据速率而乘以 2。最后，将结果除以 10910^9 以将结果转换为 GB/s。

```
有些计算使用 1024^3 而不是 10^9 进行最终计算。在这种情况下，带宽将为 836.4 GiB/s。在计算理论带宽和有效带宽时，使用相同的除数非常重要，以确保比较的有效性。
```

```
在启用 ECC 的情况下，具有 GDDR 内存的 GPU 可用 DRAM 减少了 6.25%，以存储 ECC 位。与禁用 ECC 的相同 GPU 相比，每次内存事务获取 ECC 位也会使有效带宽减少约 20%，尽管 ECC 对带宽的确切影响可能更高，并且取决于内存访问模式。另一方面，HBM2 内存提供了专用的 ECC 资源，允许无开销的 ECC 保护。
```

```
GPU ECC (Error Correcting Code) 是一种在显卡内存中使用额外位元来检测和纠正错误的技术。这种功能在某些情况下可以检测和修正严重错误，从而提高数据的可靠性和系统的稳定性。
```

### 8.2.2. Effective Bandwidth Calculation

有效带宽通过计时特定程序活动和了解程序如何访问数据来计算。为此，可以使用以下公式：

$$ \text{Effective Bandwidth} = \frac{(B_r + B_w)}{10^9 \times \text{time}} $$

其中，有效带宽的单位是 GB/s，\( B_r \) 是每个内核读取的字节数，\( B_w \) 是每个内核写入的字节数，时间以秒为单位。

例如，要计算 2048 x 2048 矩阵复制的有效带宽，可以使用以下公式：

$$ \text{Effective Bandwidth} = \frac{2048 \times 2048 \times 4 \times 2}{10^9 \times \text{time}} $$

其中，元素的数量乘以每个元素的大小（浮点数为 4 字节），再乘以 2（因为有读和写），除以 \(10^9\)（或 \(1024^3\)）以获得传输的内存 GB 数。然后将这个数除以时间（秒）以获得 GB/s。

### 8.2.3. Throughput Reported by Visual Profiler

对于计算能力为 2.0 或更高的设备，Visual Profiler 可以用来收集多种不同的内存吞吐量指标。以下吞吐量指标可以在 “Details” 或 “Detail Graphs” 视图中显示:
* Requested Global Load Throughput
* Requested Global Store Throughput
* Global Load Throughput
* Global Store Throughput
* DRAM Read Throughput
* DRAM Write Throughput

请求的全局负载吞吐量和请求的全局存储吞吐量值表示内核请求的全局内存吞吐量，因此对应于在有效带宽计算中显示的有效带宽。

由于最小内存事务大小大于大多数字长，内核所需的实际内存吞吐量可能包括内核未使用的数据传输。对于全局内存访问，实际吞吐量由全局负载吞吐量和全局存储吞吐量值报告。

需要注意的是，这两个数值都很有用。实际内存吞吐量显示了代码与硬件限制的接近程度，而有效或请求带宽与实际带宽的比较可以很好地估计由于内存访问合并不佳而浪费的带宽量（参见全局内存的合并访问）。对于全局内存访问，这种请求内存带宽与实际内存带宽的比较由全局内存负载效率和全局内存存储效率指标反映。

```
作为一个例外，HBM2 的分散写操作会受到一些来自 ECC 的开销影响，但这比在 ECC 保护的 GDDR5 内存上具有类似访问模式的开销要小得多。
```