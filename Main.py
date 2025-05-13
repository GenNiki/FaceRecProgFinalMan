import cv2
import mediapipe as mp
import numpy as np


def green_to_transparency(img, output_path, green_threshold=100):
    # Конвертация в RGBA
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Создание маски для зеленого фона (BGR)
    green_mask = (img[:, :, 1] > green_threshold) & \
                 (img[:, :, 0] < green_threshold / 2) & \
                 (img[:, :, 2] < green_threshold / 2)

    rgba[:, :, 3] = np.where(green_mask, 0, 255)
    cv2.imwrite(output_path, rgba)

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Захват видео с камеры
cap = cv2.VideoCapture(1)
square_size = 480  # Размер квадратного изображения

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    # Конвертация в RGB для MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Создание маски и основного изображения с зелёным фоном
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    output = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)
    square_output = np.full((square_size, square_size, 3), (0, 255, 0), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Получение координат ключевых точек
            landmarks = [
                (int(lm.x * width), int(lm.y * height))
                for lm in face_landmarks.landmark
            ]

            # Создание выпуклой оболочки для контура лица
            hull = cv2.convexHull(np.array(landmarks, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 255)

            # Копирование области лица на зелёный фон
            output[mask == 255] = frame[mask == 255]

            # Получение ограничивающего прямоугольника лица
            x, y, w, h = cv2.boundingRect(hull)

            # Вычисление области интереса с отступами
            padding = int(max(w, h) * 0.03)
            roi_x1 = max(0, x - padding)
            roi_y1 = max(0, y - padding)
            roi_x2 = min(width, x + w + padding)
            roi_y2 = min(height, y + h + padding)
            face_roi = output[roi_y1:roi_y2, roi_x1:roi_x2]

            if face_roi.size > 0:
                # Определение коэффициента масштабирования
                scale = square_size / max(face_roi.shape[:2])

                # Масштабирование с сохранением пропорций
                new_w = int(face_roi.shape[1] * scale)
                new_h = int(face_roi.shape[0] * scale)
                resized = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Размещение по центру квадрата
                x_offset = (square_size - new_w) // 2
                y_offset = (square_size - new_h) // 2
                square_output[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Отображение результатов
    cv2.imshow('Face Mask', output)
    cv2.imshow('Centered Face', square_output)
    green_to_transparency(square_output, 'face_square_end.png', green_threshold=100)

    cv2.imwrite('face_square_green.jpg', square_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очистка ресурсов
cap.release()
cv2.destroyAllWindows()