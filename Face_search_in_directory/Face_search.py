import os
import os.path
import pickle
import face_recognition
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def predict(X_img_path, knn_clf=None, model_path="./trained_knn_model.clf", distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Неверный путь к изображению: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Должен предоставить классификатор knn либо через knn_clf, либо через model_path")

        # Загрузить обученную модель KNN (если она была передана)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

        # Загрузите файл изображения и найдите местоположение лица
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # Если на изображении не найдены лица, вернуть пустой результат.
    if len(X_face_locations) == 0:
        return []

        # Найти кодировки для лиц в тесте iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Используйте модель KNN, чтобы найти лучшие совпадения для тестового лица
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Прогнозировать классы и удалять классификации, которые не находятся в пределах порога
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

for image_file in os.listdir("knn_examples/test"):
    full_file_path = os.path.join("knn_examples/test", image_file)

    print("Поиск лиц в {}".format(image_file))

    # Найти всех людей на изображении, используя обученную модель классификатора
    # Примечание: вы можете передать либо имя файла классификатора, либо экземпляр модели классификатора
    predictions = predict(full_file_path, model_path="trained_knn_model.clf")

    # Вывести результаты на консоль
    if predictions==[]:
        print("- @@@ - Некорректно изображено лицо на фотографии {}".format(image_file))
    for name, (top, right, bottom, left) in predictions:
            print("- Найдено {} в {}".format(name, image_file))