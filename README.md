# Mobile Device Movement Simulation

A CDC-GAN (Conditional Deep Convolutional Generative Adversarial Network) model based on [[1]](#1) and [[2]](#2). It is used for training and generating 5 second artificial accelerometer and gyroscope segments.

## Directory Structure

- `./`: Contains all needed scripts for the training of the model and subsequent generating of data.
- `data/`: Includes not processed and preprocessed sample data files used for the simulation.
- `models/`: Trained CDC-GAN models.
- `README.md`: This file.

## Getting Started

To train the model and generate data, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mobile_device_movement_simulation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mobile_device_movement_simulation
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Preprocess json data:
    ```bash
    python preprocess.py
    ```
5. Train the model:
   ```bash
   python train.py
   ```
6. Export the '.pkl' model to '.onnx' for browser use:
   ```bash
   python export.py
   ```
7. **(Optional)** Generate a single sample:
   ```bash
   python generate.py
   ```

## References

1. [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
2. [GANs for 1D Signal](https://github.com/LixiangHan/GANs-for-1D-Signal)
