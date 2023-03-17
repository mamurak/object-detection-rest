from os import environ

from object_detection import preprocess_encoded_image, detect_objects


prediction_url = environ.get('PREDICTION_URL')


def predict(body):
    base64encoded_image = body.get('image')
    image_data = preprocess_encoded_image(base64encoded_image)
    detections = detect_objects(image_data, prediction_url)
    mapped_detections = map_(*detections)

    return {'detections': mapped_detections}


def map_(boxes, scores, class_labels):
    cleaned = []
    max_boxes = 10
    num_detections = min(len(scores), max_boxes)

    for i in range(num_detections):
        box = boxes[i]
        d = {
            'box': {
                'yMin': box[0]/416,
                'xMin': box[1]/416,
                'yMax': box[2]/416,
                'xMax': box[3]/416,
            },
            'class': class_labels[i],
            'label': class_labels[i],
            'score': scores[i],
        }
        cleaned.append(d)

    return cleaned
