import torch
import torch.backends.cudnn as cudnn

from run_on_video import clip, txt2clip, vid2clip
clip_model_version = "ViT-B/32"

clip_model, _ = clip.load(clip_model_version, device=0, jit=False)
x = torch.zeros(1, 3, 224, 224).cuda()
y = clip_model.encode_image(x)
print(y.shape) # 1, 512

