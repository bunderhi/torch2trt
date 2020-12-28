# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch2trt
import matplotlib.pyplot as plt
import cv2


# %%
# Load the jetracer trained model 
model = torch.load('/models/run10/weights.pt')


# %%
model = model.cuda().eval().half()


# %%
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']


# %%
model_w = ModelWrapper(model).half()


# %%
data = torch.ones((1,3,320,640)).cuda().half()


# %%
model_trt = torch2trt.torch2trt(model_w, [data], fp16_mode=True)

# %% [markdown]
# # Live demo

# %%
ino = 289
# Read  a sample image and mask from the data-set
img = cv2.imread(f'/models/train_data/Images/{ino:03d}_cam-image1_.jpg').transpose(2,0,1).reshape(1,3,300,650)
mask = cv2.imread(f'/models/train_data/Masks/{ino:03d}_cam-image1_mask.png')
input = torch.from_numpy(img).type(torch.cuda.FloatTensor)/255


# %%
with torch.no_grad():
    output = model(input)


# %%
# Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
plt.hist(output['out'].data.cpu().numpy().flatten())


# %%
# Plot the input image, ground truth and the predicted output
plt.figure(figsize=(10,10));
plt.subplot(131);
plt.imshow(img[0,...].transpose(1,2,0));
plt.title('Image')
plt.axis('off');
plt.subplot(132);
plt.imshow(mask);
plt.title('Ground Truth')
plt.axis('off');
plt.subplot(133);
plt.imshow(output['out'].cpu().detach().numpy()[0][0]>0.4);
plt.title('Segmentation Output')
plt.axis('off');


# %%
output_trt = model_trt(input)


# %%
# Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
plt.hist(output_trt['out'].data.cpu().numpy().flatten())


# %%
# Plot the input image, ground truth and the predicted output
plt.figure(figsize=(10,10));
plt.subplot(131);
plt.imshow(img[0,...].transpose(1,2,0));
plt.title('Image')
plt.axis('off');
plt.subplot(132);
plt.imshow(mask);
plt.title('Ground Truth')
plt.axis('off');
plt.subplot(133);
plt.imshow(output_trt['out'].cpu().detach().numpy()[0][0]>0.4);
plt.title('Segmentation Output')
plt.axis('off');


