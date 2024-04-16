from os import environ

from boto3 import client


def load_model(model_path='yolov5n/1/yolov5n.onnx'):
    print('Commencing model loading.')

    s3_endpoint_url = environ.get(
        'S3_ENDPOINT_URL', environ.get('AWS_S3_ENDPOINT')
    )
    s3_access_key = environ.get(
        'S3_ACCESS_KEY', environ.get('AWS_ACCESS_KEY_ID')
    )
    s3_secret_key = environ.get(
        'S3_SECRET_KEY', environ.get('AWS_SECRET_ACCESS_KEY')
    )
    s3_bucket_name = environ.get(
        'S3_BUCKET_NAME', environ.get('AWS_S3_BUCKET')
    )

    print(f'Downloading model from bucket {s3_bucket_name} '
          f'from S3 storage at {s3_endpoint_url}')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key
    )

    s3_client.download_file(
        s3_bucket_name, f'models/{model_path}', 'model.onnx'
    )

    print('Finished model loading.')


if __name__ == '__main__':
    load_model()
