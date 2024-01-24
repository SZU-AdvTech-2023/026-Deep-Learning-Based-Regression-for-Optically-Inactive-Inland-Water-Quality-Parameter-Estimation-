本代码为高光谱水质反演的实验代码，在这次实验中，探究了不同模型变体在处理高光谱图像时的性能差异。通过对比 patchDNNR、 pixelDNNR、
带有注意力机制的 patchDNNR 和使用平滑 L1 损失函数的 patchDNNR， 我们观察了 MAE、 RMSE、 R2 等评价指标的变化趋势，并对结果进行了分析。