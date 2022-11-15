import boto3
from boto3.dynamodb.conditions import Key
import os


class Dynamodb:


    def __init__(self):
        SERVER_SECRET_KEY = os.getenv("SECRET")
        SERVER_PUBLIC_KEY = os.getenv("ACCESSKEY")
        REGION_NAME = os.getenv('REGION_NAME')
        self.dynamodb_client = dynamodb = boto3.resource('dynamodb',
                                            aws_access_key_id=SERVER_PUBLIC_KEY,
                                            aws_secret_access_key=SERVER_SECRET_KEY,
                                            region_name=REGION_NAME
                                            )
        self.statustable = self.dynamodb_client.Table('statustable')

    def getStatus(self):
        response = self.statustable.get_item(
            Key={
                'statuskey': 'personpresent'
            }
        )
        statusvalue = response['Item']['statusvalue']
        return statusvalue


    def setStatus(self,value, sessionid):
        if (value == 'True'):
            updatevalue = '{"presence": {"S": "'+ value + '"},"sessionID": {"S": "'+sessionid+'"}}'
        else:
            updatevalue = value

        self.statustable.update_item(
            Key={
                'statuskey': 'personpresent'
            },
            UpdateExpression='SET statusvalue = :statusvalue',
            ExpressionAttributeValues={
                ':statusvalue': updatevalue
            }
        )

    def setStatusek(self,value):
        items_to_delete = [value]
        with self.statustable.batch_writer() as batch:
            for item in items_to_delete:
                response = batch.put_item(Item={
                    "statuskey": item["statuskey"],
                    "statusvalue": item["statusvalue"]
                })


    def getPosture(self):
        response = self.statustable.get_item(
            Key={
                'statuskey': 'currentposture'
            }
        )
        statusvalue = response['Item']['statusvalue']
        return statusvalue


    def setPosture(self,value):
        self.statustable.update_item(
            Key={
                'statuskey': 'currentposture',
            },
            UpdateExpression='SET statusvalue = :statusvalue',
            ExpressionAttributeValues={
                ':statusvalue': value
            }
        )


    def getDeviationExceeded(self):
        response = self.statustable.get_item(
            Key={
                'statuskey': 'deviationexceeded'
            }
        )
        statusvalue = response['Item']['statusvalue']
        return statusvalue


    def setDeviationExceeded(self,value):
        self.statustable.update_item(
            Key={
                'statuskey': 'deviationexceeded',
            },
            UpdateExpression='SET statusvalue = :statusvalue',
            ExpressionAttributeValues={
                ':statusvalue': value
            }
        )

    def setTooManyPeople(self, value):
        self.statustable.update_item(
            Key={
                'statuskey': 'toomanypeople'
            },
            UpdateExpression='SET statusvalue = :statusvalue',
            ExpressionAttributeValues={
                ':statusvalue': value
            }
        )

