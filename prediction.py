import base64

import tensorflow as tf


model_dir = 'models/openimages_v4_ssd_mobilenet_v2_1'
saved_model = tf.saved_model.load(model_dir)
detector = saved_model.signatures['default']


def predict(body):
    base64img = body.get('image')
    img_bytes = base64.decodebytes(base64img.encode())
    detections = _detect(img_bytes)
    cleaned = _clean_detections(detections)

    return {'detections': cleaned}


def _detect(img):
    image = tf.image.decode_jpeg(img, channels=3)
    converted_img = tf.image.convert_image_dtype(
        image, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    num_detections = len(result["detection_scores"])

    output_dict = {
        key: value.numpy().tolist() for key, value in result.items()
    }
    output_dict['num_detections'] = num_detections

    return output_dict


def _clean_detections(detections):
    cleaned = []
    max_boxes = 10
    num_detections = min(detections['num_detections'], max_boxes)

    for i in range(0, num_detections):
        d = {
            'box': {
                'yMin': detections['detection_boxes'][i][0],
                'xMin': detections['detection_boxes'][i][1],
                'yMax': detections['detection_boxes'][i][2],
                'xMax': detections['detection_boxes'][i][3]
            },
            'class': detections['detection_class_entities'][i].decode('utf-8'),
            'label': detections['detection_class_entities'][i].decode('utf-8'),
            'score': detections['detection_scores'][i],
        }
        cleaned.append(d)

    return cleaned


def _preload_model():
    blank_jpg = tf.io.read_file('blank.jpeg')
    blank_img = tf.image.decode_jpeg(blank_jpg, channels=3)
    detector(
        tf.image.convert_image_dtype(blank_img, tf.float32)[tf.newaxis, ...]
    )


_preload_model()
