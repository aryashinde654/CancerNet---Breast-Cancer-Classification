# 🩺 CancerNet – Breast Cancer Histopathology Image Classifier

## 📋 Project Overview

**CancerNet** is a Convolutional Neural Network (CNN)-based image classification model designed to accurately classify **breast histopathology images** as **benign** or **malignant**. The model was trained on the publicly available **IDC\_regular dataset** (Invasive Ductal Carcinoma) from Kaggle, aiming to assist pathologists in early cancer detection, reduce manual diagnosis time, and minimize misdiagnosis cases.

---

## 📁 Project Structure

```
cancernet-vscode/
├── data/IDC_regular/                 # Dataset folder (Kaggle IDC_regular dataset)
├── models/                           # Trained model files (.h5)
├── outputs/                          # Confusion matrix, predictions CSV
├── src/
│   ├── dataset.py                    # Data loading & preprocessing
│   ├── model.py                      # CancerNet CNN architecture
│   ├── train.py                      # Model training script
│   └── predict.py                    # Inference / predictions script
├── requirements.txt                  # Python dependencies list
└── README.md                         # Project documentation (this file)
```

---

## 🗃️ Dataset Details

* **Dataset Name**: IDC\_regular (Breast Histopathology Images)
* **Source**: [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
* **Data Volume**: 277,524 image patches of size 50x50 pixels

  * Negative (Benign): 198,738 images
  * Positive (Malignant): 78,786 images
* **Disk Space Required**: \~3.02GB

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cancernet-vscode.git
cd cancernet-vscode
```

### 2. Setup Virtual Environment & Install Dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate    # On Windows
pip install -r requirements.txt
```

### 3. Download Dataset & Place It

* Download **IDC\_regular** dataset from Kaggle.
* Extract it inside the folder: `data/IDC_regular/`

### 4. Train the CancerNet Model

```bash
python src/train.py --data_dir data/IDC_regular --epochs 10
```

### 5. Run Inference on Test Dataset

```bash
python src/predict.py --model_path models/cancernet_best.h5 --image_dir data/IDC_regular/test --output outputs/predictions.csv
```

---

## 📊 Results

| Metric                | Value  |
| --------------------- | ------ |
| Training Accuracy     | 87.37% |
| Validation Accuracy   | 85.29% |
| Test Accuracy         | 81.23% |
| Precision (Malignant) | 80.5%  |
| Recall (Malignant)    | 78.8%  |
| F1-Score (Malignant)  | 79.6%  |

* Confusion Matrix & prediction CSV will be saved in the `outputs/` folder.
* The trained model file (`cancernet_best.h5`) will be saved in the `models/` directory.

---

## 🛠️ Technologies Used

* Python 3.10
* TensorFlow 2.19
* Keras API
* Scikit-learn
* Matplotlib, Pandas, NumPy

---

## 🔍 Future Enhancements

* Apply **Transfer Learning** with ResNet or EfficientNet architectures.
* Incorporate **advanced data augmentation** techniques.
* Deploy the model as a **web-based diagnostic tool** for real-time classification.
* Collaborate with medical institutions for real-world testing.

---

## 🙏 Acknowledgments

* **Kaggle** for providing open-access datasets.
* The **TensorFlow and Keras** teams for robust ML frameworks.
* Medical researchers and pathologists whose work inspired this project.
* OpenAI's **ChatGPT** for assisting in project design & documentation workflow.

---

## 📄 License

This project is for educational and research purposes only. Not for clinical or commercial use.

