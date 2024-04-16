import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load the model
        self.model = load_model("./model/keras_Model.h5", compile=False)

        # Load the labels
        self.class_names = open("./model/labels.txt", "r").readlines()

        # Set up the UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Create QLabel to display image
        self.label = QLabel()
        self.layout.addWidget(self.label)

        # Create QPushButton to select image
        self.button = QPushButton("Select Image")
        self.layout.addWidget(self.button)
        self.button.clicked.connect(self.select_image)

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)", options=options)
        if file_path:
            self.predict_image(file_path)

    def predict_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        processed_image = self.preprocess_image(image)
        data = np.expand_dims(processed_image, axis=0)
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        QMessageBox.information(self, "Prediction Result", f"Class: {class_name[2:]}\nConfidence Score: {confidence_score}")

    def preprocess_image(self, image):
        size = (224, 224)
        image = image.resize(size)
        normalized_image_array = (np.array(image).astype(np.float32) / 127.5) - 1
        return normalized_image_array

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
