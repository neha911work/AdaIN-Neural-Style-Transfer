# 🎨 AdaIN Neural Style Transfer

An AI-powered image stylization project that combines the content of one image with the artistic style of another using **Adaptive Instance Normalization (AdaIN)** and deep learning.

The project uses PyTorch and a VGG19-based feature extractor to generate stylized images, with a Flask web application for interacting with the model.

## ✨ Project Overview

Neural Style Transfer (NST) is a computer vision technique that applies the artistic characteristics of a style image to the content of a different image while preserving its main structure.

This project implements **Adaptive Instance Normalization (AdaIN)**, a technique that aligns the channel-wise mean and variance of content features with those of style features to transfer artistic styles efficiently.

### Key Features

- **Content and Style Image Input:** Use separate images to define the content and artistic appearance of the output.
- **AdaIN-Based Style Transfer:** Transfer visual characteristics from a style image to a content image.
- **Deep Learning with PyTorch:** Use a VGG19-based encoder and a decoder network for image stylization.
- **Flask Web Application:** Access the style-transfer workflow through a web interface.
- **GPU-Based Model Training:** Train the model using GPU acceleration in a Kaggle environment.
- **Customizable Style Strength:** Adjust the influence of the style image using the application's style-strength control, where supported by the implementation.

## 🧠 Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Computer Vision:** Torchvision
- **Model Architecture:** Adaptive Instance Normalization (AdaIN), VGG19-based encoder, decoder network
- **Web Framework:** Flask
- **Development Tools:** VS Code, Jupyter Notebook, Kaggle
- **Training Hardware:** NVIDIA Tesla T4 GPU (Kaggle)

## 🏗️ Project Structure

```text
NST_PROJECT/
│
├── models/
│   ├── decoder.pth
│   └── vgg_normalised.pth
│
├── utils/
│   ├── app.py
│   ├── model.py
│   └── utils.py
│
├── Templates/
│   └── index.html
│
├── static/
│   └── ...
│
├── notebook-nst-neha.ipynb
├── train.py
├── sample.py
├── .gitignore
└── README.md
```

**Note:** The structure above highlights the main project files. The `static/` directory may contain additional assets, stylesheets, and application resources.

## ⚙️ How It Works

1. **Input Images:** Provide a content image and a style image.
2. **Feature Extraction:** Use the VGG19-based encoder to extract feature representations from the images.
3. **Adaptive Instance Normalization:** Align the statistical properties of content features with those of style features.
4. **Image Reconstruction:** Pass the transformed features through the decoder to reconstruct the stylized image.
5. **Output:** Display the generated image through the application workflow.

## 🖥️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/neha911work/AdaIN-Neural-Style-Transfer.git
cd AdaIN-Neural-Style-Transfer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the Python packages required by the project. If a `requirements.txt` file is available, use:

```bash
pip install -r requirements.txt
```

If the repository does not include a dependency file yet, install the packages required by the project's imports and verify compatible versions of PyTorch, Torchvision, Flask, and Pillow.

### 4. Run the Flask application

From the project root directory, execute:

```bash
python -m utils.app
```

Open the local address printed in the terminal by Flask to access the web application.

## 🚀 Model Training

The model training workflow is implemented in `train.py`.

For full-dataset training, I used a Kaggle environment with GPU acceleration.

| Training detail | Configuration |
|---|---|
| Framework | PyTorch |
| GPU | NVIDIA Tesla T4 |
| Epochs | 5 |
| Batch size | 4 |
| Content dataset | COCO-style `test2017` dataset |
| Style dataset | Painter-style image dataset |
| Training environment | Kaggle |

The training workflow uses content and style images to optimize the model's feature reconstruction and style-transfer objectives.

**Important:** The original training datasets and intermediate experiment checkpoints are not included in this GitHub repository. The project uses the model weight files included in `models/` for inference.

## 📂 Model Weights

- `models/vgg_normalised.pth` — pretrained VGG19-based encoder weights used for feature extraction.
- `models/decoder.pth` — decoder weights used to reconstruct stylized images.

The model weights must be present at the paths expected by the application. If a weight file is missing, it must be obtained separately or generated through the appropriate training workflow.

## 🎯 Learning Outcomes

Through this project, I explored:

- Neural Style Transfer and Adaptive Instance Normalization.
- Deep feature extraction using VGG19.
- Model training and GPU-accelerated experimentation.
- Image processing and reconstruction using PyTorch.
- Integrating a deep learning model into a Flask web application.
- Managing model checkpoints and organizing a machine learning project for deployment.

## 🔮 Future Improvements

- Extend image stylization to video frames.
- Improve inference speed and memory efficiency.
- Support additional artistic styles and image formats.
- Deploy the application to a cloud hosting platform.
- Improve user experience with clearer progress indicators and output previews.

## 👩‍💻 Author

**Neha Jadhav**

GitHub: [@neha911work](https://github.com/neha911work)

---

*Built as a deep learning and computer vision project exploring artistic image generation using Adaptive Instance Normalization.*
