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