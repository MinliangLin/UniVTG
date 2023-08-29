# dataset test
from run_on_video.video_loader import VideoLoader
from PIL import Image

dataset = VideoLoader(
    'data/1260022902/1260022902_video.mp4',
    framerate=1/60,
    size=224,
    centercrop=False,
    overwrite=True,
)

video = dataset[0]['video']
print(dataset[0], video.shape)
for i in range(10):
    frame = video[i].permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(frame.astype('uint8'))
    pil_img.save(f'tmp/{i}_s.png')
