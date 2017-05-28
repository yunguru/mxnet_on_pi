# recyclesorter.py

import AWSIoTPythonSDK
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
from datetime import datetime
import sys
import logging
import time
import getopt
import picamera
import inception_predict
import random
import time
import boto3
import json
import pyaudio
import inflect
from time import sleep

print("start")

host = "xxxxxxxxxxxxx.iot.xx-xxxx-x.amazonaws.com" # Change this to your IoT host name
rootCAPath = "./certs/root-CA.crt"
certificatePath = "./certs/recyclesorter.cert.pem"
privateKeyPath = "./certs/recyclesorter.private.key"
deviceids = [1,2,3,4,5,6,7,8,9,10]
# List of common recylable items.  This can be changed to any object list that you are trying to identify
recycletags = ['water bottle', 'bottle', 'plastic', 'recycle', 'soda can', 'sodacan', 'pepsi', 'coke', 'soda’,’ paper', 'cup’,’ aluminum', 'newspaper’, ‘cardboard', 'milk', 'juice']
topicname = ""

region = 'xx-xxx-x' # change this to switch to another AWS region e.g. us-west-2
colors = [ ['green', 0,255,0], ['blue', 255,0,0], ['red', 0,0,255], ['purple', 255,0,255], ['silver', 192,192,192], ['cyan', 0,255,255], ['orange', 255,99,71], ['white', 255,255,255], ['black', 0,0,0] ]


polly = boto3.client("polly", region_name=region)
reko = boto3.client('rekognition', region_name=region)
p = inflect.engine()
pya = pyaudio.PyAudio()

# Functions

# Provide a string and an optional voice attribute and play the streamed audio response
# Defaults to the Salli voice
def speak(text_string, voice="Joanna"):
	try:
	    # Request speech synthesis
	    response = polly.synthesize_speech(Text=text_string,
	    	TextType="text", OutputFormat="pcm", VoiceId=voice)
	except (BotoCoreError, ClientError) as error:
	    # The service returned an error, exit gracefully
	    print(error)
	    exit(-1)
	# Access the audio stream from the response
	if "AudioStream" in response:
		stream = pya.open(format=pya.get_format_from_width(width=2), channels=1, rate=16000, output=True)
		stream.write(response['AudioStream'].read())
		sleep(1)
		stream.stop_stream()
		stream.close()
	else:
	    # The response didn't contain audio data, return False
	    print("Could not stream audio")
	    return(False)

# Amazon Rekognition label detection
def reko_detect_labels(image_bytes):
	print("Calling Amazon Rekognition: detect_labels")
#	speak("Detecting labels with Amazon Recognition")
	response = reko.detect_labels(
	    Image={
    	    'Bytes': image_bytes
    	},
    	MaxLabels=8,
    	MinConfidence=60
	)
	return response

# create verbal response describing the detected labels in the response from Rekognition
# there needs to be more than one label right now, otherwise you'll get a leading 'and'
def create_verbal_response_labels(reko_response):
	mystringverbal = "I detected the following labels: "
	mystring = ""
	humans = False
	labels = len(reko_response['Labels'])
	if labels == 0:
		mystringverbal = "I cannot detect anything."
	else:
		i = 0
		for mydict in reko_response['Labels']:
			i += 1
			if mydict['Name'] == 'People':
				humans = True
				continue
			print("%s\t(%.2f)" % (mydict['Name'], mydict['Confidence']))
			if i < labels:
				newstring = "%s, " % (mydict['Name'].lower())
				mystring = mystring + newstring
			else:
				newstring = "and %s. " % (mydict['Name'].lower())
				mystring = mystring + newstring
			if ('Human' in list(mydict.values())) or ('Person' in list(mydict.values())) :
				humans = True
	mystringverbal = mystringverbal + mystring
	return humans, mystring, mystringverbal


# end of function

# Init AWSIoTMQTTClient For Publish/Subscribe Communication With Server
myAWSIoTMQTTClient = AWSIoTMQTTClient("recycletsorter")
myAWSIoTMQTTClient.configureEndpoint(host, 8883)
myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)


# AWSIoTMQTTClient connection configuration
myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec


# Start the Camera and tell the Server we are alive
print("Running camera")
camera = picamera.PiCamera()


# Capture image
filename = '/home/pi/cap.jpg'
#camera.start_preview()
camera.capture(filename)
#camera.stop_preview()
fin = open(filename, 'rb')
encoded_image_bytes = fin.read()
fin.close()



topn = inception_predict.predict_from_local_file(filename, N=2)

# MQTT settings
topicname = 'recyclesorter-' + str(datetime.now().strftime("%Y-%m-%d-H"))
deviceid = random.choice(deviceids)
datetimenow = str(datetime.now())
myAWSIoTMQTTClient.connect()
horizonalpin = 15
vertizalpin = 12

recylefound = 0

for obj in recycletags:
    if obj in topn[0][1] or obj in topn[1][1]:
        msg = str(deviceid) + '|1|' + str(obj) + '|' + '|' + str(datetimenow)
        speakmsg = "Recyclable item detected by MXNet is " + str(obj)
        recylefound = 1
        break

if recylefound == 0:
    labels=reko_detect_labels(encoded_image_bytes)
    humans, labels_response_string, lables_verbal_response_string = create_verbal_response_labels(labels)

    msg = str(deviceid) + '|0|' + labels_response_string + '|' + str(datetimenow)
    for obj2 in recycletags:
            if obj2 in labels_response_string:
                msg = str(deviceid) + '|1|' + str(obj2) + '|' + '|' + str(datetimenow)
                speakmsg = "Recyclable item detected by Rekognition is " + str(obj2)
                recylefound = 1
                break

if recylefound == 0:
        speakmsg = "Recyclable items not detected. " + lables_verbal_response_string

print(speakmsg)
speak(speakmsg)
myAWSIoTMQTTClient.publish(topicname, msg, 0)
myAWSIoTMQTTClient.disconnect()
