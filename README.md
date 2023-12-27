# DGA

This is the official repository for “DGA”.

## SJTU-4K Dataset

<img width="489" alt="image" src="https://github.com/zytx121/DGA/assets/10410257/5711a95a-f33a-4724-bc2b-2a120da3f695">

<img width="489" alt="image" src="https://github.com/zytx121/DGA/assets/10410257/6a07b3a2-80c1-4434-a5c8-581e1965343b">



To collect the data we require, we design a reasonable scheme for data capture. First, we chose 20 scenes on our campus as our experimental site, including streets and car parks, where many kinds of cars are often seen. Afterward, we used the DJI Mini 3 drone as the capturing tool. In our scheme, the flight height ranges from 20 m to 120 m, and there were 9 flight heights in total. The resolution to the raw images is 4000 $\times$ 2260 pixels. We also provide aerial images captured at different pitch angles and directions, which are closer to real scenes. Figure 1 shows several images captured by our drone with different pitch angles from the heights of 20 m, 40 m, 80 m, and 110 m, respectively. The size of the objects is quite diverse, and the number of objects in a single image is large, especially for images captured at a great height. The detailed distribution of data in the SJTU-4K dataset is shown in Figure 2. However, this size is too large, being unsuitable for our experiment. Therefore, we need to conduct some processing of the raw images. They are tailored into a smaller size of 1024 $\times$ 1024 when being attacked. Ultimately, there are 10260 training images with 57172 car objects and 6345 testing images with 19969 car objects. To cooperate with the new patch-based attack evaluation method proposed, we used rotated boxes to label vehicles. Unlike the common rotated box representation, we use a 360-degree angle representation to retain the direction information of vehicles. 

The dataset is available now. 
[GoogleDrive](https://drive.google.com/drive/folders/1LWXC-a7OM2kGbXeCp-frrMcBOFD127Hb?usp=sharing)
[jbox](https://jbox.sjtu.edu.cn/l/j1vS3y)(提取码：jlfr)

## Code

The code is coming soon.

