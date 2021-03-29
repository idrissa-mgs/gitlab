import logging
import boto3
from botocore.exceptions import ClientError


from joblib import dump
from sklearn import svm
from sklearn import datasets

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    print("------ Uploading to s3 --------")
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
    
 
 
def train_model():
    print("---------- TRAINING --------")
    model = svm.SVC()
    X, y = datasets.load_digits(return_X_y=True)
    model.fit(X, y)
    print("Training done successfully")
    return model

def export_model(model):
    print("---------- EXPORT ---------")
    file_name = './digits_model.joblib'
    dump(model, file_name)
    upload_file(file_name=file_name, bucket = "jedha-bootcamp-idris", object_name="CI-CD"+file_name)

def start():
    model = train_model()
    export_model(model)
    print('Model successfully exported.')

if __name__ == '__main__':
    start()

