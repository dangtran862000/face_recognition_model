import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

from imutils import paths
from sklearn.metrics import accuracy_score
from face_recognition import face_locations
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "Images/"

EMBEDDING_FL = "nn4.small2.v1.t7"
DATASET_PATH = os.path.join(path, "Dataset")
IMAGE_TEST = os.path.join(path, "Vin Diesel/1.jpg")

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def load_torch(model_path_fl):
    """
      model_path_fl: Link file chứa weigth của model
      """
    model = cv2.dnn.readNetFromTorch(model_path_fl)
    return model


encoder = load_torch(EMBEDDING_FL)


def image_read(image_path):
    """
  input:
    image_path: link file ảnh
  return:
    image: numpy array của ảnh
  """
    image = cv2.imread(image_path)
    return image


def extract_bbox(image, single=True):
    """
  Trích xuất ra tọa độ của face từ ảnh input
  input:
    image: ảnh input theo kênh RGB.
    single: Lấy ra 1 face trên 1 bức ảnh nếu True hoặc nhiều faces nếu False. Mặc định True.
  return:
    bbox: Tọa độ của bbox: <start_Y>, <start_X>, <end_Y>, <end_X>
  """
    bboxs = face_locations(image)
    if len(bboxs) == 0:
        return None
    if single:
        bbox = bboxs[0]
        return bbox
    else:
        return bboxs


def extract_face(image, bbox, face_scale_thres=(20, 20)):
    """
    input:
      image: ma trận RGB ảnh đầu vào
      bbox: tọa độ của ảnh input
      face_scale_thres: ngưỡng kích thước (h, w) của face. Nếu nhỏ hơn ngưỡng này thì loại bỏ face
    return:
      face: ma trận RGB ảnh khuôn mặt được trích xuất từ image input.
    """
    h, w = image.shape[:2]
    try:
        (startY, startX, endY, endX) = bbox
    except:
        return None
    minX, maxX = min(startX, endX), max(startX, endX)
    minY, maxY = min(startY, endY), max(startY, endY)
    face = image[minY:maxY, minX:maxX].copy()
    # extract the face ROI and grab the ROI dimensions
    (fH, fW) = face.shape[:2]

    # ensure the face width and height are sufficiently large
    if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
        return None
    else:
        return face


image = image_read(IMAGE_TEST)
bbox = extract_bbox(image)
face = extract_face(image, bbox)

DATASET_PATH = "./Images"


def model_processing(face_scale_thres=(20, 20)):
    """
    face_scale_thres: Ngưỡng (W, H) để chấp nhận một khuôn mặt.
    """
    image_links = list(paths.list_images(DATASET_PATH))
    images_file = []
    y_labels = []
    faces = []
    total = 0
    for image_link in image_links:
        split_img_links = image_link.split("/")
        # Lấy nhãn của ảnh
        name = split_img_links[-2]
        # Đọc ảnh
        image = image_read(image_link)
        (h, w) = image.shape[:2]
        # Detect vị trí các khuôn mặt trên ảnh. Gỉa định rằng mỗi bức ảnh chỉ có duy nhất 1 khuôn mặt của chủ nhân classes.
        bbox = extract_bbox(image, single=True)
        # print(bbox_ratio)
        if bbox is not None:
            # Lấy ra face
            face = extract_face(image, bbox, face_scale_thres=(20, 20))
            if face is not None:
                faces.append(face)
                y_labels.append(name)
                images_file.append(image_links)
                total += 1
            else:
                next
    print("Total bbox face extracted: {}".format(total))
    return faces, y_labels, images_file


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def blobImage(image, out_size=(300, 300), scaleFactor=1.0, mean=(104.0, 177.0, 123.0)):
    """
      input:
        image: ma trận RGB của ảnh input
        out_size: kích thước ảnh blob
      return:
        imageBlob: ảnh blob
      """
    # Chuyển sang blobImage để tránh ảnh bị nhiễu sáng
    imageBlob = cv2.dnn.blobFromImage(image,
                                      scalefactor=scaleFactor,  # Scale image
                                      size=out_size,  # Output shape
                                      mean=mean,  # Trung bình kênh theo RGB
                                      swapRB=False,  # Trường hợp ảnh là BGR thì set bằng True để chuyển qua RGB
                                      crop=False)
    return imageBlob


def embedding_faces(encoder, faces):
    emb_vecs = []
    for face in faces:
        faceBlob = blobImage(face, out_size=(96, 96), scaleFactor=1 / 255.0, mean=(0, 0, 0))
        # Embedding face
        encoder.setInput(faceBlob)
        vec = encoder.forward()
        emb_vecs.append(vec)
    return emb_vecs


def _most_similarity(embed_vecs, vec, labels):
    sim = cosine_similarity(embed_vecs, vec)
    sim = np.squeeze(sim, axis=1)
    argmax = np.argsort(sim)[::-1][:1]
    label = [labels[idx] for idx in argmax][0]
    return label


faces, y_labels, images_file = model_processing()

save_pickle(faces, "./faces.pkl")
save_pickle(y_labels, "./y_labels.pkl")
save_pickle(images_file, "./images_file.pkl")

embed_faces = embedding_faces(encoder, faces)
# Nhớ save embed_faces vào Dataset.
save_pickle(embed_faces, "./embed_blob_faces.pkl")
embed_faces = load_pickle("./embed_blob_faces.pkl")
y_labels = load_pickle("./y_labels.pkl")

ids = np.arange(len(y_labels))

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(np.stack(embed_faces), y_labels, ids,
                                                                       test_size=0.2, stratify=y_labels)
X_train = np.squeeze(X_train, axis=1)
X_test = np.squeeze(X_test, axis=1)
print(X_train.shape, X_test.shape)
print(len(y_train), len(y_test))

save_pickle(id_train, "./id_train.pkl")
save_pickle(id_test, "./id_test.pkl")

# Lấy ngẫu nhiên một bức ảnh trong test
vec = X_test[1].reshape(1, -1)
# Tìm kiếm ảnh gần nhất
_most_similarity(X_train, vec, y_train)

y_preds = []
for vec in X_test:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(X_train, vec, y_train)
    y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))


def base_network():
    model = VGG16(include_top=True, weights=None)
    dense = Dense(128)(model.layers[-4].output)
    norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense)
    model = Model(inputs=[model.input], outputs=[norm2])
    return model


model = base_network()
model.summary()

faces = load_pickle("./faces.pkl")
faceResizes = []
for face in faces:
    face_rz = cv2.resize(face, (224, 224))
    faceResizes.append(face_rz)

X = np.stack(faceResizes)
print(X.shape)

id_train = load_pickle("./id_train.pkl")
id_test = load_pickle("./id_test.pkl")

X_train, X_test = X[id_train], X[id_test]

print(X_train.shape)
print(X_test.shape)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss())

gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(22)
history = model.fit(
    gen_train,
    steps_per_epoch=50,
    epochs=10)

X_train_vec = model.predict(X_train)
X_test_vec = model.predict(X_test)

y_preds = []
for vec in X_test_vec:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(X_train_vec, vec, y_train)
    y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

no_batch = 0
X_au = []
y_au = []
for i in np.arange(len(X_train)):
    no_img = 0
    for x in datagen.flow(np.expand_dims(X_train[i], axis=0), batch_size=1):
        X_au.append(x[0])
        y_au.append(y_train[i])
        no_img += 1
        if no_img == 5:
            break

model2 = base_network()

model2.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss())

# Điều chỉnh tăng batch_size = 64
gen_train2 = tf.data.Dataset.from_tensor_slices((X_au, y_au)).repeat().shuffle(1024).batch(22)
history = model2.fit(
    gen_train2,
    steps_per_epoch=50,
    epochs=20)

data_tf = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    horizontal_flip=True
)

data_tf.fit(X_test)

no_batch = 0
X_test_tf = []
for i in np.arange(len(X_test)):
    no_img = 0
    for x in data_tf.flow(np.expand_dims(X_test[i], axis=0), batch_size=1):
        X_test_tf.append(x[0])
        no_img += 1
        if no_img == 1:
            break

X_train_vec = model2.predict(np.stack(X_au))
X_test_vec = model2.predict(np.stack(X_test_tf))

y_preds = []
for vec in X_test_vec:
    vec = vec.reshape(1, -1)
    y_pred = _most_similarity(X_train_vec, vec, y_au)
    y_preds.append(y_pred)

print(accuracy_score(y_preds, y_test))


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################


def _normalize_image(image, epsilon=0.000001):
    means = np.mean(image.reshape(-1, 3), axis=0)
    stds = np.std(image.reshape(-1, 3), axis=0)
    image_norm = image - means
    image_norm = image_norm / (stds + epsilon)
    return image_norm


IMAGE_OUTPUT = "./predictions.jpg"
IMAGE_PREDICT = "./test1.jpg"

# Trích xuất bbox image
image = image_read(IMAGE_PREDICT)
# imageBlob = _blobImage(image)
bboxs = extract_bbox(image, single=False)
# print(len(bboxs))
faces = []
for bbox in bboxs:
    face = extract_face(image, bbox, face_scale_thres=(20, 20))
    # face = face.copy()
    faces.append(face)
    try:
        face_rz = cv2.resize(face, (224, 224))
        # Chuẩn hóa ảnh bằng hàm _normalize_image
        face_tf = _normalize_image(face_rz)
        face_tf = np.expand_dims(face_tf, axis=0)
        # Embedding face
        vec = model2.predict(face_tf)
        # Tìm kiếm ảnh gần nhất
        name = _most_similarity(X_train_vec, vec, y_au)
        # Tìm kiếm các bbox
        (startY, startX, endY, endX) = bbox
        minX, maxX = min(startX, endX), max(startX, endX)
        minY, maxY = min(startY, endY), max(startY, endY)
        pred_proba = 0.891
        text = "{}: {:.2f}%".format(name, pred_proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (minX, minY), (maxX, maxY), (0, 0, 255), 2)
        cv2.putText(image, text, (minX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except:
        print("Not found face")
cv2.imwrite(IMAGE_OUTPUT, image)


plt.figure(figsize=(16, 8))
img = plt.imread(IMAGE_OUTPUT)
plt.imshow(img)
