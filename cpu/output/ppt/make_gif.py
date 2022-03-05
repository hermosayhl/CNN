import imageio
import os
import cv2

dir_name = '7'

img_paths = [os.path.join(dir_name, "{}.png".format(i)) for i in range(1, 4)]



img_paths = sorted(img_paths, key=lambda x: int(os.path.split(x)[0]))
print(img_paths)

gif_images = []
for path in img_paths:
    gif_images.append(imageio.imread(path))
    gif_images[-1] = cv2.resize(gif_images[-1], (1280, 720))
imageio.mimsave("{}.gif".format(dir_name),gif_images,fps=1)
