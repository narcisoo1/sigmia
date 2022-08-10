import torch
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(torch.cpu.device_count())