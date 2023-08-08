from os import environ

from classes import classes
from object_detection import detect_objects
from preprocessing import preprocess_encoded_image


prediction_url = environ.get('PREDICTION_URL')
class_labels_type = environ.get('CLASS_LABELS_TYPE', 'coco')
class_labels = classes[class_labels_type]


def predict(body):
    base64encoded_image = body.get('image')
    transformed_image, scaling, padding = preprocess_encoded_image(
        base64encoded_image
    )
    detections = detect_objects(
        transformed_image, prediction_url, len(class_labels)
    )
    mapped_detections = map_(detections, class_labels)

    return {'detections': mapped_detections}


def map_(objects, class_labels, edge_length=640):
    cleaned = []

    for object_ in objects:
        d = {
            'box': {
                'yMin': float(object_[1]/edge_length),
                'xMin': float(object_[0]/edge_length),
                'yMax': float(object_[3]/edge_length),
                'xMax': float(object_[2]/edge_length),
            },
            'class': class_labels[int(object_[5])],
            'label': class_labels[int(object_[5])],
            'score': float(object_[4]),
        }
        cleaned.append(d)

    return cleaned
