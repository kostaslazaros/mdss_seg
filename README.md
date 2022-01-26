# Medical Image Segmentation using Unet and DoubleUnet

This is a comparative study between Unet and DoubleUnet on medical image segmentation. For the purposes of this study, the CVC dataset was used (contains colonoscopy images and their corresponding segmented masks).

# How to

- Clone this project

- Download CVC dataset [here](https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0) and extract it in the project's root folder.

- Convert images from .tif format to .jpeg format (use a program like irfanview).

- Download pretrained DoubleUnet model for CVC dataset [here](https://drive.google.com/file/d/14ahqFsLu-XlW8IRYmYptVocRGwsGm6Ea/view?usp=sharing)

- Create a virtual environment and run:

  `pip install -r requirements.txt`

# Citations

- [doubleUnet original project by Debesh Jha, PhD researcher working on medical image analysis using deep learning.](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net)

- [Unet implementation by zhixuhao](https://github.com/zhixuhao/unet)

- [Unet implementation for polyp segmentation by Nikhil Tomar, AI Researcher, YouTuber, Blogger](https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0)

# Special Thanks To

- Dr. Dimitris Iakovidis: Professor, Dept of Computer Science and Biomedical Informatics, University of Thessaly, Lamia, Greece.

- Dimitra-Christina Koutsiou: Ph.D. Student, Dept of Computer Science and Biomedical Informatics, University of Thessaly, Lamia, Greece.
