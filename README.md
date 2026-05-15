# ALL Snow Removed: Single Image Desnowing Algorithm Using Hierarchical Dual-tree Complex Wavelet Representation and Contradict Channel Loss <br> (Accepted by ICCV'21)


![image](folder/result.png)

# Dataset
We also propose a large scale dataset called Comprehensive Snow Dataset (CSD). It can present the snow scenes in more comprehensive way. You can leverage this dataset to train your network.<br>
[[Dataset Download]](https://drive.google.com/file/d/1smNrDvtPs89e0xk336Rt2-2KZffyX5H-/view?usp=sharing)
![image](folder/csd.png)

Training
```
python ./train.py --logPath ./your_log_path --dataPath /path_to_data/data.npy --gtPath /path_to_gt/gt.npy --batchsize batchsize --epochs epochs --modelPath ./path_to_exist_model/model_to_load.h5 --validation_num number_of_validation_image --steps_per_epoch steps_per_epoch
```

*data.npy should be numpy of training image whose shape is (number_of_image, 480, 640, 3). The range is (0, 255) and the datatype is uint8 or int.<br>
*gt.npy should be numpy of ground truth image, whose shape is (number_of_image, 480, 640, 3). The range is (0, 255) and datatype is uint8 or int.

Example:
```
python ./train.py --logPath ./log --dataPath ./training_data.npy --gtPath ./training_gt.npy --batchsize 3 --epochs 1500 --modelPath ./previous_log/preivious_model.h5 --validation_num 200 --steps_per_epoch 80
```



Testing
```
$python predict.py -dataroot C:\Users\jason\Desktop\FYP\code\CSD\Test\Snow -predictpath C:\Users\jason\Desktop\FYP\code\CSD\tf_predict -batch_size 1
```
*datatype default: tif, jpg ,png

Examples
```
$ 
python ./predict.py -dataroot ./testImg -predictpath ./p -batch_size 3
python ./predict.py -dataroot ./testImg -datatype tif -predictpath ./p -batch_size 3
```


The pre-trained model can be downloaded from [here](https://drive.google.com/file/d/1ILUWgrBPFaaDlq67YlBZYxJEbF-qZbgu/view?usp=sharing). 
Put the "finalmodel.h5" to the 'modelParam'.

# Newest Thing to Notice
If u want to try this in torch version please run command like: 
```
$ 
python train_torch.py  --dataPath C:\Users\jason\Desktop\FYP\code\CSD\Train\data.npy   --gtPath C:\Users\jason\Desktop\FYP\code\CSD\Train\gt.npy   --logPath C:\Users\jason\Desktop\FYP\code\CSD\logs --batchsize 1  --epochs 50  --steps_per_epoch 20
```
 Then using the model to make some work.