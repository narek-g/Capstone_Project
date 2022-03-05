# MLE_bootcamp_capstone
2021 December Cohort 

A semantic segmentation model is developed using U2-Net, trained on the imaterilaist dataset. 

More on u2-Net: https://xuebinqin.github.io/U2Net_PR_2020.pdf 

iMaterialist dataset: https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data

The model is hosted on an AWS server. 
It may be accessed by either visiting http://18.144.74.125/docs, or executing a curl command through ther terminal (as shown in https://github.com/narek-g/Capstone_Project/blob/main/Production/Deployment/deployment.ipynb). 

The model works best if an image of less than 400 kB is sent. 