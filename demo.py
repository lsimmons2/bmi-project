
import cv2
import sys
import dlib
import numpy as np
from contextlib import contextmanager
import urllib2
from model import get_model
import config




def get_trained_model():
    weights_file = 'bmi_model_weights.h5'
    model = get_model(ignore_age_weights=True)
    model.load_weights(weights_file)
    return model


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images_from_camera():
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, img = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture image")
            yield img


def run_demo():
    args = sys.argv[1:]
    multiple_targets = '--multiple' in args
    single_or_multiple = 'multiple faces' if multiple_targets else 'single face'
    model = get_trained_model()
    print 'Loading model to detect BMI of %s...' % single_or_multiple

    NUMBER_OF_FRAMES_IN_AVG = 20
    last_seen_bmis = []
    detector = dlib.get_frontal_face_detector()

    for img in yield_images_from_camera():
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1)
        faces = np.empty((len(detected), config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - config.MARGIN * w), 0)
                yw1 = max(int(y1 - config.MARGIN * h), 0)
                xw2 = min(int(x2 + config.MARGIN * w), img_w - 1)
                yw2 = min(int(y2 + config.MARGIN * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH)) / 255.00

            predictions = model.predict(faces)

            if multiple_targets:
                for i, d in enumerate(detected):
                    label = str(predictions[i][0])
                    draw_label(img, (d.left(), d.top()), label)
            else:
                last_seen_bmis.append(predictions[0])
                if len(last_seen_bmis) > NUMBER_OF_FRAMES_IN_AVG:
                    last_seen_bmis.pop(0)
                elif len(last_seen_bmis) < NUMBER_OF_FRAMES_IN_AVG:
                    continue
                avg_bmi = sum(last_seen_bmis) / float(NUMBER_OF_FRAMES_IN_AVG)
                label = str(avg_bmi)
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow('result', img)
        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == '__main__':
    run_demo()
