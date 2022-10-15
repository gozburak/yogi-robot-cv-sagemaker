import boto3

class Dynamodb:

    dynamodb = None
    statustable = None

    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        self.statustable = self.dynamodb.Table('statustable')

    def getStatus(self):
        response = self.statustable.get_item(
            Key={
                'statuskey': 'personpresent'
            }
        )
        statusvalue = response['Item']['statusvalue']
        return statusvalue


    def setStatus(self,value):
        self.statustable.update_item(
            Key={
                'statuskey': 'personpresent'
            },
            UpdateExpression='SET statusvalue = :statusvalue',
            ExpressionAttributeValues={
                ':statusvalue': value
            }
        )


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
