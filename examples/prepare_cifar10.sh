#!/bin/bash

mojo package mimage/mimage -o mimage.mojopkg
wget http://pjreddie.com/media/files/cifar.tgz
tar xvf cifar.tgz -C ./examples/data/
