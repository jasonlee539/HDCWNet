import os
import cv2
import numpy as np

snow_dir = r"C:\Users\jason\Desktop\FYP\code\CSD\Train\Snow"
gt_dir   = r"C:\Users\jason\Desktop\FYP\code\CSD\Train\Gt"

save_data_path = r"C:\Users\jason\Desktop\FYP\code\CSD\Train\data.npy"
save_gt_path   = r"C:\Users\jason\Desktop\FYP\code\CSD\Train\gt.npy"

snow_imgs = []
gt_imgs = []

files = sorted(os.listdir(snow_dir))

for name in files:
    snow_path = os.path.join(snow_dir, name)
    gt_path = os.path.join(gt_dir, name)

    if not os.path.exists(gt_path):
        print("找不到对应GT:", gt_path)
        continue

    snow = cv2.imread(snow_path)
    gt = cv2.imread(gt_path)

    if snow is None or gt is None:
        print("读取失败:", name)
        continue

    snow = cv2.resize(snow, (640, 480))
    gt = cv2.resize(gt, (640, 480))

    snow_imgs.append(snow)
    gt_imgs.append(gt)

data = np.array(snow_imgs, dtype=np.uint8)
gt = np.array(gt_imgs, dtype=np.uint8)

print("data shape:", data.shape)
print("gt shape:", gt.shape)

np.save(save_data_path, data)
np.save(save_gt_path, gt)

print("保存完成")