import torch
# from archs import DEUNeXt
from archs_light import DEUNeXt

from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 创建resnet50网络
model = DEUNeXt(num_classes=1)

# 创建输入网络的tensor
tensor = (torch.rand(1, 3, 224, 224),)

# 分析FLOPs
flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total())

# 分析parameters
print(parameter_count_table(model))

