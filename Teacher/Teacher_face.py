import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}



def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []
    summa = 0
    for filename in os.listdir(path='knn_examples/train'):
        summa += 1
    # Проходить через каждого человека в тренировочном наборе
    put = 0
    for class_dir in os.listdir(train_dir):
        put += 1
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            print("пройдена {}/{} пути)".format(put, summa))
            continue

            # Цикл каждого учебного изображения для текущего человека

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):

            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # Если в тренировочном образе нет людей (или слишком много людей), пропустите изображение.
                if verbose:
                    print("Изображение {} не подходит для обучения: {}".format(img_path, "Не найдено лицо" if len(face_bounding_boxes) < 1 else "Найдено более одного лица"))

            else:
                # Добавить кодировку лица для текущего изображения в тренировочный набор
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
        print("пройдена {}/{} пути)".format(put, summa))


        # Определить, сколько соседей использовать для взвешивания в классификаторе KNN
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("ВыберАНО n_neighbors автоматически:", n_neighbors)

        # Создать и обучить классификатор KNN
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Сохранить обученный классификатор KNN
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

if __name__ == "__main__":
    # ШАГ 1. Обучите классификатор KNN и сохраните его на диск
    # После того, как модель обучена и сохранена, вы можете пропустить этот шаг в следующий раз.
    print("Учебный классификатор КНН ...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Обучение завершено!")

