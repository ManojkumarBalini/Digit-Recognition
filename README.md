Here is your **FULL copy-paste README.md** (no breaks, no extra text — just copy everything 👇)

---

````markdown
# IP2: Digit Recognition with CNN

This project implements a simple Convolutional Neural Network (CNN) for handwritten digit recognition using PyTorch and Lightning. The model is trained on grayscale 28×28 digit images.

---

## 📌 Prerequisites

- Python 3.8 or later
- Required libraries:
  - torch
  - torchvision
  - lightning
  - numpy
  - matplotlib
  - click

---

## ⚙️ Installation

(Optional) Create a virtual environment and install dependencies:

```bash
pip install torch torchvision lightning numpy matplotlib click
````

---

## 📁 Project Structure

```
IP2_Template/
├── data/
│   └── img_data/               # Contains dataset files
├── src/
│   ├── cnn_model.py            # CNN model architecture
│   └── written_digit_cnn.py    # Training script
├── README.md                   # Project documentation
└── ip2_report_<wpi-username>.pdf   # Final report
```

---

## 📊 Dataset

Place the `.npy` dataset files inside:

```
data/img_data/
```

Required files:

* train_images.npy
* train_labels.npy
* val_images.npy
* val_labels.npy

👉 Image shape: `(N, 1, 28, 28)`
👉 Pixel values: `[0–255]` or normalized `[0–1]`

---

## 🚀 Training the Model

Run the following command from the project root:

```bash
python src/written_digit_cnn.py --data-dir data/img_data --epochs 10 --save-plot
```

---

### 🔧 What Happens During Training

* Model trains for given epochs
* Best model saved as `cnn.ckpt`
* Plots generated (if enabled):

  * `loss_vs_step.png`
  * `val_acc_vs_epoch.png`
* Outputs validation accuracy and confusion matrix

---

## 🧠 Command-Line Arguments

| Argument       | Default           | Description         |
| -------------- | ----------------- | ------------------- |
| --data-dir     | ../data/img_data/ | Dataset path        |
| --batch-size   | 64                | Training batch size |
| --epochs       | 10                | Number of epochs    |
| --lr           | 1e-3              | Learning rate       |
| --weight-decay | 1e-5              | L2 regularization   |
| --ckpt         | ./cnn.ckpt        | Checkpoint path     |
| --save-plot    | False             | Save training plots |

---

### ✅ Example (Custom Training)

```bash
python src/written_digit_cnn.py --data-dir data/img_data --batch-size 128 --lr 0.001 --epochs 15 --save-plot
```

---

## 📈 Outputs

* ✔ Model checkpoint: `cnn.ckpt`
* ✔ Training loss plot
* ✔ Validation accuracy plot
* ✔ Console logs with metrics

---

## 🤖 Autograder Compatibility

This project is compatible with autograder systems.

Expected command:

```bash
python src/written_digit_cnn.py --data-dir <path> --epochs 10 --ckpt <path> --no-save-plot
```

⚠️ Do NOT change:

* Class names
* Function signatures

Autograder uses:

```
SmallCNN (from cnn_model.py)
```

## 📌 Notes

* Keep dataset structure unchanged
* Ensure correct file paths
* Use proper hyperparameters for better accuracy

---

## 📜 License

This project is for educational purposes only.
