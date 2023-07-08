import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def get_label_from_folder_name(folder_name):
    if folder_name == "folder1" or folder_name == "Test1":
        return "5 TL"
    elif folder_name == "folder2" or folder_name == "Test2":
        return "10 TL"
    elif folder_name == "folder3" or folder_name == "Test3":
        return "20 TL"
    elif folder_name == "folder4" or folder_name == "Test4":
        return "50 TL"
    elif folder_name == "folder5" or folder_name == "Test5":
        return "100 TL"
    elif folder_name == "folder6" or folder_name == "Test6":
        return "200 TL"
    elif folder_name == "folder7" or folder_name == "Test7":
        return "1 kr"
    elif folder_name == "folder8" or folder_name == "Test8":
        return "5 kr"
    elif folder_name == "folder9" or folder_name == "Test9":
        return "10 kr"
    elif folder_name == "folder10" or folder_name == "Test10":
        return "25 kr"
    elif folder_name == "folder11" or folder_name == "Test11":
        return "50 kr"
    elif folder_name == "folder12" or folder_name == "Test12":
        return "1 TL"
    else:
        return None


def extract_edge_features(image):
    # Image okuma ve boyutlandırma
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))

    # Kenarları algılama (Canny edge detection)
    edges = cv2.Canny(img, 100, 200)

    # Kenar özelliklerini tek boyutlu diziye çevirme
    edge_features = edges.flatten()

    return edge_features

def extract_sift_features(image):
    # Image okuma ve boyutlandırma
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))

    # Gri tona çevirme
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # SIFT özellik çıkarımı yapma
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # SIFT descriptor'larını tek boyutlu diziye indirgeme
    # ve ilk 128 descriptor'ı alma
    if descriptors is not None:
        sift_features = descriptors.flatten()
        sift_features = sift_features[:128]
    else:
        sift_features = np.zeros(128)

    return sift_features

def extract_shape_features(image):
    # Image okuma ve boyutlandırma
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))

    # Gri tona çevirme
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Kenarları algılama (Canny edge detection)
    edges = cv2.Canny(gray, 100, 200)

    # Yalnızca en dıştaki çizgileri(konturları) alma
    contours, _ = cv2.findContours(edges,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Kontur sayısını elde etme
    shape_features = np.array([len(contours)])

    return shape_features

def extract_color_features(image):
    # Image okuma ve boyutlandırma
    img = cv2.imread(image)
    img = cv2.resize(img, (256, 256))

    # Renk dönüşümü yapma (RGB'den HSV uzayına)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Renk histogramını hesaplama
    hist = cv2.calcHist([hsv_img],
                        [0, 1], None,
                        [180, 256],
                        [0, 180, 0, 256])

    # Histogramı tek boyuta indirgeme
    color_features = hist.flatten()

    return color_features

def train_knn_model(features, labels):
    # KNN modelini oluşturma ve eğitme
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, labels)

    return knn


# Eğitim veri setini okuma
train_folder_path = "D:\\4_2-Bahar-22_23\\Pattern Recognition\\Final Project\\MoneyClassification\\TrainingImages"
train_images = []
train_labels = []

for folder_name in os.listdir(train_folder_path):
    folder_path = os.path.join(train_folder_path, folder_name)
    if os.path.isdir(folder_path):
        label = get_label_from_folder_name(folder_name)
        if label is not None:
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                train_images.append(image_path)
                train_labels.append(label)

# Eğitim veri setinden fature çıkarımı
train_edge_features = []
train_sift_features = []
train_shape_features = []
train_color_features = []

for img in train_images:
    train_edge_features.append(extract_edge_features(img))
    train_sift_features.append(extract_sift_features(img))
    train_shape_features.append(extract_shape_features(img))
    train_color_features.append(extract_color_features(img))

# SIFT descripto'larını yeniden şekillendirme (boyut hatasını gidermek için)
train_sift_features = [np.reshape(features, (-1, 128)) if len(features) > 0 else np.zeros((1, 128)) for features in train_sift_features]

# Tüm özellikleri birleştirme
train_features = np.concatenate((train_edge_features, np.vstack(train_sift_features), np.array(train_shape_features).reshape(-1, 1), train_color_features), axis=1)

# LabelEncoder oluşturma (özellikle AUC-ROC için ama gerek olmayabilir)
label_encoder = LabelEncoder()

# Etiketleri sayısal değerlere dönüştürme
encoded_train_labels = label_encoder.fit_transform(train_labels)

# KNN modelini eğitme
knn_model = train_knn_model(train_features, encoded_train_labels)

# Test veri setini yükleme
test_folder_path = "D:\\4_2-Bahar-22_23\\Pattern Recognition\\Final Project\\MoneyClassification\\TestImages"
test_images = []
test_labels = []

for folder_name in os.listdir(test_folder_path):
    folder_path = os.path.join(test_folder_path, folder_name)
    if os.path.isdir(folder_path):
        label = get_label_from_folder_name(folder_name)
        if label is not None:
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                test_images.append(image_path)
                test_labels.append(label)

# Test veri setinin feature'larını çıkartma
test_edge_features = []
test_sift_features = []
test_shape_features = []
test_color_features = []

for img in test_images:
    test_edge_features.append(extract_edge_features(img))
    test_sift_features.append(extract_sift_features(img))
    test_shape_features.append(extract_shape_features(img))
    test_color_features.append(extract_color_features(img))

# SIFT descriptor'larını yeniden şekillendirme
test_sift_features = [np.reshape(features, (-1, 128)) if len(features) > 0 else np.zeros((1, 128)) for features in test_sift_features]

# Tüm özellikleri birleştirme
test_features = np.concatenate((test_edge_features, np.vstack(test_sift_features), np.array(test_shape_features).reshape(-1, 1), test_color_features), axis=1)

# Etiketleri sayısal değerlere dönüştürme
encoded_test_labels = label_encoder.transform(test_labels)

# Test veri seti üzerinde KNN modelinin tahminler yapması
predictions = knn_model.predict(test_features)

# Confusion Matrix oluşturma
cm = confusion_matrix(encoded_test_labels, predictions)

# Confusion Matrix ayarları
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Plot ayarları
tick_marks = np.arange(len(np.unique(test_labels)))
plt.xticks(tick_marks, np.unique(test_labels), rotation=45)
plt.yticks(tick_marks, np.unique(test_labels))

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Confusion Matrix en sonda plot edilmiştir.

# Accuracy hesaplama
accuracy = accuracy_score(encoded_test_labels, predictions)
print("Accuracy:", accuracy)

# F1-score hesaplama
f_score = f1_score(encoded_test_labels, predictions, average='weighted')
print("F-Score:", f_score)

# Recall hesaplama
recall = recall_score(encoded_test_labels, predictions, average='macro')
print("Recall:", recall)

# Precision hesaplama
precision = precision_score(encoded_test_labels, predictions, average='macro')
print("Precision:", precision)


# Feature Importance Analizi
feature_names = ['Edge Features', 'SIFT Features', 'Shape Features', 'Color Features']
feature_importances = []

# Her feature'ın model doğruluğuna etkisini analiz etmek için özellikleri tek tek çıkartma
for i in range(len(feature_names)):
    # i. özelliği çıkarma
    reduced_train_features = np.delete(train_features, i, axis=1)

    f_knn_model = train_knn_model(reduced_train_features, encoded_train_labels)

    reduced_test_features = np.delete(test_features, i, axis=1)
    f_predictions = f_knn_model.predict(reduced_test_features)

    f_accuracy = accuracy_score(encoded_test_labels, f_predictions)
    feature_importances.append(f_accuracy)

# Feature importance'ları normalize etme
feature_importances = np.array(feature_importances) / np.sum(feature_importances)

# Feature Importance Results
for feature, importance in zip(feature_names, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")

# CM görüntüleme
plt.show()