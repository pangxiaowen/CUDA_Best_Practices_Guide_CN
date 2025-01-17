# 7. Optimizing CUDA Applications
每轮应用程序并行化完成后，开发人员可以着手优化实现以提高性能。由于可以考虑的优化方案很多，深入理解应用程序的需求可以帮助使这个过程尽可能顺利。然而，正如 APOD 整体一样，程序优化是一个迭代过程（识别优化机会，应用并测试优化，验证所取得的加速，重复），这意味着程序员在看到良好的加速效果之前不必花大量时间记住所有可能的优化策略。相反，可以在学习过程中逐步应用策略。

优化可以在不同级别上应用，从数据传输与计算的重叠到微调浮点操作序列。可用的分析工具在指导这一过程时非常有价值，因为它们可以帮助开发人员优化工作并提供与本指南优化部分相关的参考。