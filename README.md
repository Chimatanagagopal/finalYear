# PROJECT explanation

EYE DISEASE CLASSIFICATION USING CNN IN MATLAB

This MATLAB-based deep learning project focuses on the automatic classification of various eye diseases using Convolutional Neural Networks (CNN). The model was trained to recognize and differentiate between six common eye conditions and normal cases using fundus image datasets.

[Confusion Matrix image: Screenshot (12).png]

--------------------------------------------------------------------

PROJECT OVERVIEW

The goal of this project is to build a robust image classification model using MATLAB's Deep Learning Toolbox to detect and classify:

- Age-related Macular Degeneration (AMD)
- Cataract
- Diabetic Retinopathy (DR)
- Glaucoma
- Hypertension-related eye conditions
- Myopia
- Normal eye images

--------------------------------------------------------------------

MODEL ARCHITECTURE AND TRAINING

Framework           : MATLAB R2024b  
Model Type          : Custom Convolutional Neural Network (CNN)  
Validation Accuracy : 82.69%  
Epochs              : 2  
Total Iterations    : 1566  
Learning Rate       : 0.0005  
Hardware Used       : Single CPU  

--------------------------------------------------------------------

RESULTS

- Macro F1-Score: 0.7901  
- High classification accuracy for Diabetic Retinopathy and Normal classes  
- Some misclassification observed between Glaucoma and Hypertension  
- Confusion matrix and training metrics demonstrate robust performance

--------------------------------------------------------------------

DIRECTORY STRUCTURE

Eye-Disease-Classifier/
├── README.md  
├── Code/  
│   ├── transcnn.m  
│   ├── preprocessing.m  
│   └── utils.m  
├── Dataset/  
│   └── <images classified by folder>  
├── Results/  
│   └── confusion_matrix.png  

--------------------------------------------------------------------

HOW TO RUN

1. Clone the repository  
2. Open MATLAB R2024b  
3. Open and run 'transcnn.m'  
4. Ensure dataset paths are correctly set  
5. Training and evaluation will start automatically  

--------------------------------------------------------------------

FEATURES

- Real-time accuracy and loss tracking  
- Confusion matrix output for result analysis  
- Training statistics reporting  
- Includes preprocessing and data augmentation techniques  

--------------------------------------------------------------------

REQUIREMENTS

- MATLAB R2023b or later  
- Deep Learning Toolbox  
- Image Processing Toolbox  

--------------------------------------------------------------------

FUTURE ENHANCEMENTS

- Add Grad-CAM for model explainability  
- Train using GPU to improve efficiency  
- Develop a GUI with MATLAB App Designer for deployment  

--------------------------------------------------------------------

AUTHOR

Naga Gopal Chimata  
Engineer in Electronics & Communication | Deep Learning Enthusiast

--------------------------------------------------------------------

REFERENCE

This work relates to advanced techniques published in "Expert Systems With Applications" (Elsevier, 2024) for behavioral biometrics and medical image classification.  
DOI: 10.1016/j.eswa.2023.122808

--------------------------------------------------------------------

LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
