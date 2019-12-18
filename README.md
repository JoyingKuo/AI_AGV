# Deep learning for agv

這是based on Python 3 及 Keras 所實作的AGV自走車專案。 該模型會對每張影像中的軌道特徵進行分析，並輸出行走方向以及速度。

* 角錐道路辨識
![Cone road](https://i.imgur.com/GPaVwyJ.png)

* 天花板軌道辨識
![Celling road](https://i.imgur.com/xNil0Ug.png)



```
left_wheel_speed: 左輪速度

left_wheel_dir:左輪方向(1為正轉,0為反轉),括號中的數值為模型實際輸出的數值

right_wheel_speed: 右輪速度

left_wheel_dir:右輪方向(1為正轉,0為反轉),括號中的數值為模型實際輸出的數值
```


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
   * 請將要訓練的資料包成**資料夾**的格式存到偏好的路徑，意即開啟`data/train_data/`不能直接看到`.png`檔案
      ![](https://i.imgur.com/RW3p1kM.png)
   
  
   * 檔案名稱的格式為`年-月-日-時-分-秒-id-左輪方向(p/n)左輪速-右輪方向(p/n)右輪速`    
      ![](https://i.imgur.com/p0VS1eJ.png)
   
    
4. 資料前處理
   * configure 檔案路徑
    ```
        cd code/preprocess/
        vim preprocess.py
    ```
   * 修改以下內容至符合你專案所需
    ![](https://i.imgur.com/7yKjipI.png)


   * 執行前處理
    `python preprocess.py`
      * 會輸出訓練列表，提供給step 5 訓練用
    ![](https://i.imgur.com/a8Zg1aa.png)
    
     ```
        第一欄: index
        第二欄: left_wheel_dir（左輪方向，1正轉，0反轉）
        第三欄: left_wheel_speed（左輪速度）
        第四欄: right_wheel_dir（右輪方向，1正轉，0反轉）
        第五欄: right_wheel_speed（右輪速度）
        第六欄: filename(訓練圖片的檔案路徑)
      ```

    
5. 訓練agv
   * configure 檔案路徑
    ```
    cd code/train/
    vim train.py
    ```
   * 修改以下內容至符合你專案所需
    ![](https://i.imgur.com/PyWf6J0.png)

   * 執行訓練
   `python train.py`
      * 會輸出神經網路模型，提供給step 6 評估用、 step 7現場試驗用
        ![](https://i.imgur.com/aZa2BuY.png)
    
        ```
        檔案名稱的格式為 `loss_儲存時間.h5`
        ```


6. 評估訓練結果
   * configure 檔案路徑
     ```
        cd code/test/
        vim predict_result.py
      ```
    * 修改以下內容至符合你專案所需
    ![](https://i.imgur.com/YHeHr84.png)
    
    * 執行評估
       `python predict_result.py`
       * 評估單張圖片（output_mode = 0）
         * 會顯示單一照片的評估結果
            ![](https://i.imgur.com/tZtQVcK.png)
       * 評估整個影片（output_mode = 1）
         * 會輸出原始影片(i.e origin.mp4)以及評估結果影片(i.e.predict.mp4)

    
7. 現場試驗
    * configure 檔案路徑
    ```
    cd code/test/
    vim predict_agv.py
    ```
    * 修改以下內容至符合你專案所需
    ![](https://i.imgur.com/p4J6iK2.png)

 
    * 執行現場試驗
    `python predict_agv.py`



