# import cv2


# img = cv2.imread(r"C:\Users\jason\Desktop\FYP\code\CSD\tf_predict\320326.jpg")
# img_resize = cv2.resize(
#     img,
#     (640, 360),
#     interpolation=cv2.INTER_CUBIC
# )


# cv2.imwrite("output_640x360.jpg", img_resize)

# print("done")

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

img1 = cv2.imread(r"output_640x360.jpg")
img2 = cv2.imread(r"A:\carla\PythonAPI\examples\data\320327.png")

img1 = cv2.resize(img1, (640, 360))
img2 = cv2.resize(img2, (640, 360))

mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
psnr_value = psnr(img2, img1, data_range=255)
ssim_value = ssim(img2, img1, channel_axis=2, data_range=255)

print("MSE:", mse)
print("PSNR:", psnr_value)
print("SSIM:", ssim_value)