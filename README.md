# Deep learning for agv

這是based on Python 3 及 Keras 所實作的AGV自走車專案。 該模型會對每張影像中的軌道特徵進行分析，並輸出行走方向以及速度。
* 場景- 角錐道路
![](https://i.imgur.com/GPaVwyJ.png =80%x)

* 場景- 天花板軌道
![](https://i.imgur.com/xNil0Ug.png =80%x)



:::info
left_wheel_speed: 左輪速度

left_wheel_dir:左輪方向(1為正轉,0為反轉),括號中的數值為模型實際輸出的數值

right_wheel_speed: 右輪速度

left_wheel_dir:右輪方向(1為正轉,0為反轉),括號中的數值為模型實際輸出的數值
:::


這個專案包括:
* Code for pre-processing
* Code for training
* Code for evaluation
* Code for runtime detection




## Requirements
OS 環境 [Ubuntu 16.04](http://releases.ubuntu.com/16.04/) and [TensorFlow-Release 19.08](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_19.08.html#rel_19.08), 還有其他會使用到的package都列在 `install.sh`.


## Execution Flow
1. 下載這個專案並解壓縮
    `git clone https://github.com/joyingKuo/AI_AGV`
2. 安裝相依套件
   ```bash
    cd AI_AGV/
    bash code/install.sh
   ```
3. 上傳訓練<span style="color:red;">資料夾</span>到你想要的訓練路徑，`i.e. data/train_data/`
![](https://i.imgur.com/RW3p1kM.png =80%x)

    
    :::warning
    請將要訓練的資料包成<span style="color:red;">資料夾</span>的格式存到偏好的路徑，意即開啟`data/train_data/`不能直接看到`.png`檔案
    :::
   ![](https://i.imgur.com/p0VS1eJ.png =80%x)

    :::warning
    檔案名稱的格式為
    `年-月-日-時-分-秒-id-左輪方向(p/n)左輪速-右輪方向(p/n)右輪速`
    :::
4. 資料前處理
    4-1. configure 檔案路徑
    ```
    cd code/preprocess/
    vim preprocess.py
    ```
    修改以下內容至正確路徑
    ![](https://i.imgur.com/LUKLUs6.png)
    
    :::info
    path:訓練資料夾路徑(會讀取該資料夾底下的所有資料夾)
    csv_path: 訓練列表的存放路徑
    csv_filename:訓練列表的路徑(包含檔名)
    :::

    
    4-2. 執行前處理
    `python preprocess.py`
    * 會輸出訓練列表，提供給step 5 訓練用
    ![](https://i.imgur.com/a8Zg1aa.png)
    
        :::info
        第一欄: index
        第二欄: left_wheel_dir（左輪方向，1正轉，0反轉）
        第三欄: left_wheel_speed（左輪速度）
        第四欄: right_wheel_dir（右輪方向，1正轉，0反轉）
        第五欄: right_wheel_speed（右輪速度）
        第六欄: filename(訓練圖片的檔案路徑)
        :::

    
5. 訓練agv
   5-1. configure 檔案路徑
   ```
    cd code/train/
    vim pyramid_train.py
    ```
    修改以下內容至正確路徑
    ![](https://i.imgur.com/OfzHgFQ.png)

    
    :::info
    csv_file_name:訓練列表的路徑
    model_path: 訓練模型的存放路徑
    :::
   5-2. configure model儲存頻率
   ![](https://i.imgur.com/nCPcFTZ.png)
   :::info
    nb_epoch_count=30 代表每30個epoch就會儲存一次model，整個訓練資料集被迭代一次叫做一個epoch
   :::

   5-3. 執行訓練
   `python pyramid_train.py`
    * 會輸出神經網路模型，提供給step 6 評估用、 step 7現場試驗用
    ![](https://i.imgur.com/aZa2BuY.png =80%x)
    
        :::info
        檔案名稱的格式為 `loss_儲存時間.h5`
        :::


6. 評估訓練結果
   6-1 configure 檔案路徑
    * 評估單張圖片
       ```
        cd code/test/
        vim predict.py
       ```
       修改以下內容至正確路徑
      ![](https://i.imgur.com/lu7bPsv.png)

        :::info
        model_name:訓練模型的路徑
        image_path: 要評估的圖片路徑
        :::
    * 評估整個影片
       ```
        cd code/test/
        vim predict_video.py
       ```
      修改以下內容至正確路徑
      ![](https://i.imgur.com/uLQJJwy.png)


        :::info
        path : 要評估的圖片路徑（資料格式與訓練資料夾一樣，將所有要評估的圖片資料夾放在這個路徑下）
        
        model_name : 訓練模型的路徑
        :::
   
   
   6-2 執行評估
    * 評估單張圖片
    `python predict.py`
    
        -    會輸出單一照片的評估結果,輸出路徑為`out.jpg`
            ![](https://i.imgur.com/tZtQVcK.png)

    
    * 評估整個影片
    `python predict_video.py`
        -    會輸出整個影片的評估結果，輸出路徑為`./predict.mp4`

    
7. 現場試驗
    7-1. configure 檔案路徑
    ```
    cd code/test/
    vim predict_video.py
    ```
    修改以下內容至正確路徑
    ![](https://i.imgur.com/zM8hKw4.png)

    
    :::info
    model_name : 欲用來現場試驗的模型路徑
    :::

    
    7-2. 執行現場試驗
    `python predict_agv.py`



