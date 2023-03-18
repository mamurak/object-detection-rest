from os import environ, path

from boto3 import resource


def ingest_data(data_folder='./data', max_images=5):
    print('Commencing data ingestion.')

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

    print(f'Downloading data from bucket "{s3_bucket_name}" '
          f'from S3 storage at {s3_endpoint_url}')

    s3 = resource(
        's3', endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_key
    )
    bucket = s3.Bucket(s3_bucket_name)

    download_count = 0
    for s3_object in bucket.objects.all():
        if download_count == max_images:
            break
        key = s3_object.key
        if key.endswith('.jpg'):
            local_file_path = path.join(data_folder, key)
            bucket.download_file(key, local_file_path)
            download_count += 1

    print('Finished data ingestion.')


if __name__ == '__main__':
    ingest_data(data_folder='/data')
