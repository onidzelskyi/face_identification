#!/usr/bin/python

from utils import FaceRecognition

fr = FaceRecognition()
fr.process('../test/g3.jpg', '../test/s3.jpg')