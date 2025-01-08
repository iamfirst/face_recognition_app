# -*- coding: utf-8 -*-

import sys
import os
import logging
from turtle import distance
import warnings
import threading
from queue import Queue
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QListWidgetItem, QListWidget, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
from Ui_face import *

envpath = os.path.expanduser('~/miniconda3/envs/yolov8/lib/python3.9/site-packages/cv2/qt/plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 人脸向量数据文件
FACE_EMBEDDINGS_FILE = 'face_embeddings.npy'

# 人脸图片目录
FACE_IMAGES_DIR = 'face_images'

# 人脸识别APP类
class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化 ArcFace 模型
        # app = FaceAnalysis(providers=['CUDAExecutionProvider'])  # 使用 GPU，若无 GPU 可移除 providers 参数
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))
        self.app = app

        # 注册的人脸数据库 (用字典存储人脸名称和特征向量)
        self.username = None
        self.detected_embedding = None
        self.face_embeddings = {}
        self.load_face_embeddings()

        # Initialize variables
        self.image = None
        self.video_capture = None
        self.timer = QTimer(self)  
        # self.timer.timeout.connect(self.update_frame)
        self.current_image = None
        
        # Initialize UI
        self.init_ui()

        # get face list
        self.get_face_list()

    def init_ui(self):
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.ui.registerButton.setEnabled(False)
        self.ui.recognitionButton.setEnabled(False)  
        self.ui.uploadButton.clicked.connect(self.upload_image)
        self.ui.startCameraButton.clicked.connect(self.start_camera)
        self.ui.registerButton.clicked.connect(self.register_face)
        self.ui.recognitionButton.clicked.connect(self.recognition_face)
        self.ui.tabWidget.setCurrentIndex(0)

    # 加载人脸数据face_embeddings.npy
    def load_face_embeddings(self):
        # 文件不存在时，创建一个空的字典
        if not os.path.exists(FACE_EMBEDDINGS_FILE):
            np.save(FACE_EMBEDDINGS_FILE, {})
        self.face_embeddings = np.load(FACE_EMBEDDINGS_FILE, allow_pickle=True).item()

    def upload_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if file_name:
                image = cv2.imread(file_name)
                if image is None:
                    raise ValueError("Failed to load image")
                
                # Store current frame for saving
                self.current_image = image.copy()
                
                # Convert to QImage and display
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                # Scale the image to fit the display while maintaining aspect ratio
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(self.ui.displayLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.ui.displayLabel.setPixmap(scaled_pixmap)
                
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            QMessageBox.information(None, "Warning", f"Failed to load image: {str(e)}")

        # 获取当前人脸向量数据
        faces = self.app.get(self.current_image)
        faces_empty = len(faces) == 0
        if faces_empty:
            QMessageBox.information(None, "Warning", "No face detected!")
            self.detected_embedding = None
        else:
            self.detected_embedding = faces[0].embedding
        # 注册和比对按钮状态
        self.ui.registerButton.setEnabled(faces_empty == False)
        self.ui.recognitionButton.setEnabled(faces_empty == False)    

    # 注册新的人脸
    def register_face(self):
        if self.current_image is None:
            QMessageBox.information(None, "Warning", "Please upload an image first.")
            return
        name = self.ui.userName.text()
        if not name:
            QMessageBox.information(None, "Warning", "Please enter a name.")
            return

        # 检查是否存在
        is_face_embedding_exist, similarity, exist_name = self.cosine_similarity()
        if is_face_embedding_exist:
            QMessageBox.information(None, "Warning", f"Face already registered. Name called {exist_name}")
            return
            
        # 保存更新后的 .npy 文件
        self.face_embeddings[name] = self.detected_embedding
        np.save(FACE_EMBEDDINGS_FILE, self.face_embeddings)
        print(f"Registered {name}, similarity is {similarity}")

        # 把人脸图片保存到FACE_IMAGES_DIR目录下，图片以name变量命名
        if not os.path.exists(FACE_IMAGES_DIR):
            os.makedirs(FACE_IMAGES_DIR)
        cv2.imwrite(os.path.join(FACE_IMAGES_DIR, f"{name}.jpg"), self.current_image)
        self.get_face_list()


    # 人脸比对
    def recognition_face(self):
        is_face_embedding_exist, similarity, exist_name = self.cosine_similarity()
        if is_face_embedding_exist:
            QMessageBox.information(None, "Warning", f"Face is found! Name called {exist_name}, similarity is {similarity}")
            return
        QMessageBox.information(None, "Warning", "No Face data!")
    
    # 余弦相似度算法, 计算两个向量之间的余弦相似度
    def cosine_similarity(self, threshold=0.6):
        if self.face_embeddings != {}:
            for name, registered_embedding in self.face_embeddings.items():
                similarity = cosine_similarity([registered_embedding], [self.detected_embedding])[0][0]
                if similarity > threshold:
                    return True, similarity, name
            
        return False, None, None

    def start_camera(self):
        print("start_camera")

    # 利用QListWidget显示图片列表
    def get_face_list(self):
        self.ui.faceList.clear()
        # 读取FACE_IMAGES_DIR目录下的图片
        images = os.listdir(FACE_IMAGES_DIR)

        # 遍历图片列表
        for image in images:
            # 创建QListWidgetItem对象，并设置图标和它的描述文字
            icon = QIcon()
            icon.addPixmap(QPixmap(os.path.join(FACE_IMAGES_DIR, image)), QIcon.Normal, QIcon.Off)
            item = QListWidgetItem(icon, image)
            # 把item添加到listWidget中
            self.ui.faceList.addItem(item)


def main():
    try:
        app = QApplication(sys.argv)
        window = FaceRecognitionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()