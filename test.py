import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# 初始化 ArcFace 模型
# app = FaceAnalysis(providers=['CUDAExecutionProvider'])  # 使用 GPU，若无 GPU 可移除 providers 参数
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# 注册的人脸数据库 (用字典存储人脸名称和特征向量)
registered_faces = {}

# 注册新的人脸
def register_face(img_path, name):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if faces:
        embedding = faces[0].embedding
        registered_faces[name] = embedding
        print(f"Registered {name}")
    else:
        print(f"No face detected in {img_path}")

# 对比人脸
def recognize_face(img_path):
    img = cv2.imread(img_path)
    faces = app.get(img)
    if not faces:
        print(f"No face detected in {img_path}")
        return

    embedding = faces[0].embedding
    max_sim = 0
    recognized_name = None

    for name, reg_embedding in registered_faces.items():
        sim = cosine_similarity([embedding], [reg_embedding])[0][0]
        print(f"Similarity with {name}: {sim:.2f}")
        if sim > max_sim:
            max_sim = sim
            recognized_name = name

    return recognized_name, max_sim

# 示例：注册人脸
register_face('test_images/image1.jpg', 'Person1')
register_face('test_images/image2.jpg', 'Person2')

# 示例：识别人脸
test_img_path = 'test_images/test_image.jpg'
recognized_name, similarity = recognize_face(test_img_path)
if recognized_name:
    print(f"Recognized: {recognized_name}, Similarity: {similarity:.2f}")
else:
    print("No matching face found")
