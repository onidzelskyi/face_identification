# face_identification
Identify a person from group of people

# Python

Python 3.5

## install

```bash
sudo pip install virtualenvwrapper
export WORKON_HOME=~/Envs
source /opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin/virtualenvwrapper.sh
mkvirtualenv -p python3.5 dev_face
sudo pip3 install Flask
sudo pip3 install -U flask-cors
```

## Prerequisities

```bash
sudo apt-get install python-pip libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk python-matplotlib
```

## TODO
add tests from google photo drive
https://drive.google.com/folderview?id=0B9BZI-1wnGRbVmlTb1V2dXY4UFE&usp=sharing

https://photos.google.com/share/AF1QipM4gGhRXgEuhuCU9mD-B4uGXMAeXwmrAExS127s5KuI0JC_124uExFFMZLOk9nOjg?key=OUhYX0hqUGVsY05TSGN0b0tqM2xGS1NXQUpENG5B

[![Coverage Status](https://coveralls.io/repos/onidzelskyi/face_identification/badge.svg?branch=master&service=github)](https://coveralls.io/github/onidzelskyi/face_identification?branch=master)
[![Build Status](https://travis-ci.org/onidzelskyi/face_identification.svg)](https://travis-ci.org/onidzelskyi/face_identification)

## Bugs
script don't check if server avaiable (e.g. at port 5000). Nedd handle 'Failed to load resource: The Internet connection appears to be offline.'

script not check if both images were uploaded

script not remove previously processed data

script not save user's data and results of the processing
