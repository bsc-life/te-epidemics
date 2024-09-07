#!/usr/bin/env bash

SCRIPT_FOLDER=$(realpath $(dirname $0))
BASE_FOLDER=$(realpath $SCRIPT_FOLDER/..)
DATA_FOLDER=$BASE_FOLDER/data

echo "Moving into data folder"
cd $DATA_FOLDER

echo "Downloading datasets from Zenodo"
wget https://zenodo.org/api/records/12207212/files-archive

echo "Renaming file to .zip and uncompressing"
mv files-archive files-archive.zip
unzip files-archive.zip
rm files-archive.zip

echo "Done!"
cd ..
