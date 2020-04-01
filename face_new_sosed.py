"""
Это пример использования алгоритма k-ближайших соседей (KNN) для распознавания лиц.
Когда я должен использовать этот пример?
Этот пример полезен, когда вы хотите узнать большой набор известных людей,
и сделать прогноз для неизвестного человека в допустимое время вычислений.
Описание алгоритма:
Классификатор knn сначала обучается на наборе помеченных (известных) лиц, а затем может предсказать человека
в неизвестном изображении путем нахождения k наиболее похожих лиц (изображения с закрытыми чертами лица под евклидовым расстоянием)
в своем обучающем наборе и выполняя большинство голосов (возможно, взвешенных) на своем ярлыке.
Например, если k = 3, и три ближайших изображения лица к данному изображению в обучающем наборе являются одним изображением Байдена
и два изображения Обамы. Результатом будет «Обама».
* В этой реализации используется взвешенное голосование, так что голоса ближайших соседей взвешиваются сильнее.
Использование:
1. Подготовьте набор изображений известных людей, которых вы хотите узнать. Организуйте изображения в одном каталоге
   с подкаталогом для каждого известного человека.
2. Затем вызовите функцию «поезд» с соответствующими параметрами. Убедитесь, что передали «model_save_path», если вы
   хотите сохранить модель на диск, чтобы вы могли повторно использовать модель, не переучивая ее.
3. Назовите «предсказать» и передайте свою обученную модель, чтобы распознать людей в неизвестном образе.
ПРИМЕЧАНИЕ: этот пример требует установки scikit-learn! Вы можете установить его с помощью pip:
$ pip3 install scikit-learn
"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
     Обрабатывает классификатор k-ближайших соседей для распознавания лиц.
    : param train_dir: каталог, который содержит подкаталог для каждого известного человека с его именем.
     (Посмотрите в исходном коде, чтобы увидеть пример структуры дерева train_dir)
     Структура:
        <Train_dir> /
        Person── <человек1> /
        So ├── <somename1> .jpeg
        So ├── <somename2> .jpeg
        │ ├── ...
        Person── <человек2> /
        So ├── <somename1> .jpeg
        So └── <somename2> .jpeg
        └── ...
    : param model_save_path: (необязательно) путь для сохранения модели на диске
    : param n_neighbors: (необязательно) количество соседей, которые будут взвешиваться в классификации. Выбирается автоматически, если не указано
    : param knn_algo: (необязательно) базовая структура данных для поддержки knn.default - ball_tree
    : param verbose: многословие обучения
    : return: возвращает классификатор knn, который был обучен по заданным данным.
    """
    X = []
    y = []

    # Проходить через каждого человека в тренировочном наборе
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
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


def predict(X_img_path, knn_clf=None, model_path="./trained_knn_model.clf", distance_threshold=0.6):
    """
    Распознает лица на заданном изображении, используя обученный классификатор KNN
    : param X_img_path: путь к распознаваемому изображению
    : param knn_clf: (необязательно) объект классификатора knn. если не указан, должен быть указан model_save_path.
    : param model_path: (необязательно) путь к засоленному классификатору knn. если не указано, то model_save_path должен быть knn_clf.
    : param distance_threshold: (необязательно) порог расстояния для классификации лица. чем оно больше, тем больше шансов
           ошибочно классифицировать неизвестного человека как известного.
    : return: список имен и местоположений лиц для распознанных лиц на изображении: [(имя, ограничительная рамка), ...].
        Для лиц непризнанных лиц имя «неизвестно» будет возвращено.
    """
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


def show_prediction_labels_on_image(img_path, predictions):
    """
    Показывает результаты распознавания лица визуально.
    : param img_path: путь к распознаваемому изображению
    : предсказания параметров: результаты функции предсказания
    :возвращение:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Нарисуйте рамку вокруг лица, используя модуль Pillow
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # В Pillow есть ошибка, из-за которой текст не в формате UTF-8
        # при использовании растрового шрифта по умолчанию
        name = name.encode("UTF-8")

        # Нарисуйте метку с именем под лицом
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Удалите библиотеку рисунков из памяти в соответствии с документами Pillow
    del draw

    # Показать полученное изображение
    pil_image.show()


if __name__ == "__main__":
    # ШАГ 1. Обучите классификатор KNN и сохраните его на диск
    # После того, как модель обучена и сохранена, вы можете пропустить этот шаг в следующий раз.
    print("Учебный классификатор КНН ...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Обучение завершено!")

    # ШАГ 2: Используя обученный классификатор, сделайте прогноз для неизвестных изображений
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Поиск лиц в {}".format(image_file))

        # Найти всех людей на изображении, используя обученную модель классификатора
        # Примечание: вы можете передать либо имя файла классификатора, либо экземпляр модели классификатора
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Вывести результаты на консоль
        for name, (top, right, bottom, left) in predictions:
            print("- Найдено {} в ({}, {})".format(name, left, top))

        # Отображение результатов наложения на изображение
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)