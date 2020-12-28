
import torch
import torch2trt
import cv2
import time

model = torch.load('/models/run10/weights.pt')
model = model.cuda().eval()

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

model_w = ModelWrapper(model)
data = torch.ones((1,3,320,640)).cuda()
print("start parse...")
model_trt = torch2trt.torch2trt(model_w,[data],fp16_mode=True, max_batch_size=1)
print("end parse...")
ino = 699
# Read  a sample image and mask from the data-set
img = cv2.imread(f'/models/train_data/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,640)
mask = cv2.imread(f'/models/train_data/Masks/{ino:03d}_mask.png')
input = torch.from_numpy(img).type(torch.cuda.FloatTensor)/255

torch.cuda.current_stream().synchronize()
t0 = time.time()

with torch.no_grad():
    output = model(input)

torch.cuda.current_stream().synchronize()

t1 = time.time()
lap = t1 - t0
print("Torch: ",lap)

pt_pred = output.cpu().detach().numpy()[0][0]>0.4
print("Torch Shape",pt_pred.shape)
torch.cuda.current_stream().synchronize()
t0 = time.time()

output_trt = model_trt(input)

t1 = time.time()
lap = t1 - t0
print("TRT: ",lap)

trt_pred = output_trt.cpu()[0][0]>0.4

print("TRT Shape",trt_pred.shape)

# compute max error
max_error = 0

max_error = torch.max(torch.abs(output - output_trt))

print("Model diff",max_error)

