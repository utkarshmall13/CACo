#!/bin/bash
# example for sample training on a very small 10k sized dataset

DATA_DIR=../data
SRC_DIR=../src
DATA_FILE=../data/caco10k.tar.gz
CACO_DIR=../data/clean_10k_geography
if [ ! -d "$DATA_DIR" ]; then
	mkdir $DATA_DIR
fi

if [ ! -f "$DATA_FILE" ]; then
	cd $DATA_DIR
	wget https://research.cs.cornell.edu/caco/data/caco/caco10k.tar.gz
	cd $SRC_DIR
fi

if [ ! -d "$CACO_DIR" ]; then
	cd $DATA_DIR
	echo "Untarring... may take a while"
	tar -xf caco10k.tar.gz
	cd $SRC_DIR
fi

echo "running python script"
