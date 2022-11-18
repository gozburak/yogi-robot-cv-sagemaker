# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#https://aws.amazon.com/premiumsupport/knowledge-center/iot-core-publish-mqtt-messages-python/
from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder
import time as t
import json
from python_settings import settings
import settings as my_local_settings

class AvaMQTTHelper:
    def __init__(self):
        if not settings.configured:
            settings.configure()
            settings.configure(my_local_settings)  # configure() receivesge a python module
        assert settings.configured  # now you are set
        self.DETECTOR = settings.DETECTOR
        # Define ENDPOINT, CLIENT_ID, PATH_TO_CERTIFICATE, PATH_TO_PRIVATE_KEY, PATH_TO_AMAZON_ROOT_CA_1, MESSAGE, TOPIC, and RANGE
        self.ENDPOINT = settings.ENDPOINT
        self.PATH_TO_PRIVATE_KEY = settings.KEY
        self.PATH_TO_CERTIFICATE = settings.CERT
        self.PATH_TO_ROOT_CERTIFICATE = settings.ROOTCERT
        self.TOPIC = settings.TOPIC
        self.CLIENT_ID = settings.CLIENT_ID
        #self.PATH_TO_AMAZON_ROOT_CA_1 = "./root-CA.crt"
        self.mqtt_connection = None
        self.resetMQTTConnection()
        self.counter = 0

    def resetMQTTConnection(self):
        try:
            if (self.mqtt_connection != None):
                disconnect_future = self.mqtt_connection.disconnect()
                disconnect_future.result()
        except:
            #ignore exception - this means that the connection was
            #already interrupted
            exception = "ignore this"
        event_loop_group = io.EventLoopGroup(1)
        host_resolver = io.DefaultHostResolver(event_loop_group)
        client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
        self.mqtt_connection = mqtt_connection_builder.mtls_from_path(
                    endpoint=self.ENDPOINT,
                    cert_filepath=self.PATH_TO_CERTIFICATE,
                    pri_key_filepath=self.PATH_TO_PRIVATE_KEY,
                    client_bootstrap=client_bootstrap,
                    #ca_filepath=self.PATH_TO_AMAZON_ROOT_CA_1,
                    ca_filepath=self.PATH_TO_ROOT_CERTIFICATE,
                    client_id=self.CLIENT_ID,
                    clean_session=False,
                    keep_alive_secs=6
                    )
         # Make the connect() call
        connect_future = self.mqtt_connection.connect()
        # Future.result() waits until a result is available
        connect_future.result()
        print("Connected!")

    def publishMessage(self, message):
        try:
            self.mqtt_connection.publish(topic=self.TOPIC, payload=message, qos=mqtt.QoS.AT_LEAST_ONCE)
            print("Published: '" + json.dumps(message) + "' to the topic: " + self.TOPIC)
        except:
            self.resetMQTTConnection()
            self.mqtt_connection.publish(topic=self.TOPIC, payload=message, qos=mqtt.QoS.AT_LEAST_ONCE)

