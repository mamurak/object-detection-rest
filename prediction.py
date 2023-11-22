from os import environ
from pprint import pformat

from yaml import load, SafeLoader

from classes import default_class_labels
from object_detection import detect_objects
from preprocessing import preprocess_encoded_image


prediction_url = environ.get('PREDICTION_URL')
raw_class_labels = environ.get('CLASS_LABELS')
class_labels = (
    load(raw_class_labels, Loader=SafeLoader)
    if raw_class_labels else default_class_labels
)
print(f'Using class labels: {pformat(class_labels)}')


def predict(body):
    print('Received prediction request.')
    base64encoded_image = body.get('image')
    transformed_image, scaling, padding = preprocess_encoded_image(
        base64encoded_image
    )
    detections = detect_objects(
        transformed_image, prediction_url, len(class_labels)
    )
    mapped_detections = map_(detections, class_labels, scaling, padding)
    payload = {'detections': mapped_detections}

    print(f'Prediction complete. Returning payload: {pformat(payload)}')
    return payload


def map_(objects, class_labels, scaling, padding, edge_length=640):
    cleaned = []

    for object_ in objects:
        x0, y0, x1, y1 = object_[:4] / scaling

        x0 -= padding[0]
        y0 -= padding[1]
        x1 -= padding[0]
        y1 -= padding[1]

        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, edge_length)
        y1 = min(y1, edge_length)

        d = {
            'box': {
                'yMin': y0 / edge_length,
                'xMin': x0 / edge_length,
                'yMax': y1 / edge_length,
                'xMax': x1 / edge_length,
            },
            'class': class_labels[int(object_[5])],
            'label': class_labels[int(object_[5])],
            'score': float(object_[4]),
        }
        cleaned.append(d)

    return cleaned
