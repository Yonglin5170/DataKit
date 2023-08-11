#!/bin/bash
dir_name=CodeDeploy/pipeline_data/models
name="*.md5"
# name=".*\.onnx"

find $dir_name -type f -name $name -exec rm {} \;
