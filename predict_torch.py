import os
import sys
import cv2
import torch
import argparse
import numpy as np

from model.torch_model import build_DTCWT_model


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Predict")

    parser.add_argument(
        "--dataroot",
        type=str,
        default="./testImg",
        help="测试图片文件夹路径；如果 datatype=npy，则这里是 npy 文件路径"
    )

    parser.add_argument(
        "--datatype",
        type=str,
        default="jpg,png,tif",
        help="图片类型，例如 jpg,png,tif；如果是 npy，则写 npy"
    )

    parser.add_argument(
        "--predictpath",
        type=str,
        default="./predictImg",
        help="输出图片保存路径"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--modelPath",
        type=str,
        default="./logs/modelBest.pth",
        help="PyTorch 模型权重路径"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    return parser.parse_args()


def progress(count, total, status=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = "|" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write("[%s] %s%s ...%s\r" % (bar, percents, "%", status))
    if count != total:
        sys.stdout.flush()
    else:
        print()


def load_images(args):
    select_names = []

    if args.datatype.lower() == "npy":
        print("Load from npy:", args.dataroot)
        data = np.load(args.dataroot)

        for i in range(data.shape[0]):
            select_names.append(str(i) + ".jpg")

    else:
        data = []
        img_types = [x.lower() for x in args.datatype.split(",")]

        print("Read img from:", args.dataroot)
        fnames = os.listdir(args.dataroot)
        print("Len of files:", len(fnames))

        count = 1
        for f in fnames:
            progress(count, len(fnames), "Loading data...")
            count += 1

            suffix = f.split(".")[-1].lower()
            if suffix not in img_types:
                continue

            img_path = os.path.join(args.dataroot, f)
            tmp = cv2.imread(img_path)

            if tmp is None:
                print("读取失败:", img_path)
                continue

            select_names.append(f)

            if tmp.shape[1] < tmp.shape[0]:
                tmp = np.rot90(tmp)

            if tmp.shape[0] != 480 or tmp.shape[1] != 640:
                tmp = cv2.resize(tmp, (640, 480), interpolation=cv2.INTER_CUBIC)

            data.append(tmp)

        data = np.array(data, dtype=np.uint8)

    print("data shape:", data.shape)
    return data, select_names


def preprocess(data):
    print("Start padding and convert BGR to YCrCb")

    processed = []

    for i in range(data.shape[0]):
        progress(i + 1, data.shape[0], "Padding and converting...")

        img = data[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        img = np.pad(
            img,
            ((16, 16), (16, 16), (0, 0)),
            mode="constant"
        )

        img = img.astype(np.float32) / 255.0

        # NHWC -> NCHW
        img = np.transpose(img, (2, 0, 1))

        processed.append(img)

    processed = np.array(processed, dtype=np.float32)
    print("processed shape:", processed.shape)

    return processed


def load_model(args):
    print("---------- Build Model ----------")

    model = build_DTCWT_model((3, 512, 672))
    model = model.to(args.device)

    ckpt = torch.load(args.modelPath, map_location=args.device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print("Loaded model:", args.modelPath)
    return model


@torch.no_grad()
def predict(model, data, args):
    preds = []

    total = data.shape[0]

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)

        batch = data[start:end]
        batch = torch.from_numpy(batch).float().to(args.device)

        output = model(batch)

        if isinstance(output, tuple) or isinstance(output, list):
            output = output[0]

        output = torch.clamp(output, 0.0, 1.0)

        # NCHW -> NHWC
        output = output.detach().cpu().numpy()
        output = np.transpose(output, (0, 2, 3, 1))

        preds.append(output)

        progress(end, total, "Predicting...")

    preds = np.concatenate(preds, axis=0)
    return preds


def save_results(preds, select_names, args):
    if not os.path.exists(args.predictpath):
        os.makedirs(args.predictpath)

    print("Save output")

    for i in range(preds.shape[0]):
        progress(i + 1, preds.shape[0], "Saving output...")

        img = np.clip(preds[i], 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

        # YCrCb -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

        if args.datatype.lower() == "npy":
            save_name = str(i) + ".jpg"
        else:
            save_name = os.path.splitext(select_names[i])[0] + ".jpg"

        save_path = os.path.join(args.predictpath, save_name)
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    args = parse_args()

    print("Device:", args.device)

    data, select_names = load_images(args)
    data = preprocess(data)

    model = load_model(args)

    preds = predict(model, data, args)

    save_results(preds, select_names, args)

    print("Predict finished.")