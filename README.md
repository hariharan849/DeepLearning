# DeepLearning
Basics of Deep Learning in image processing

# Building Detectron2 from Source

1. **Clone the repository**:
    ```sh
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Build and install Detectron2**:
    ```sh
    python setup.py build develop
    ```

4. **Verify the installation**:
    ```sh
    python -c "import detectron2; print(detectron2.__version__)"
    ```
