#!/bin/bash

dir_name=/dataset/yonglinwu/SMore/DataKit/.recycle_bin
type="f"
name="*.py"

if [ $type == "f" ];
then
    find $dir_name -type $type -name $name -exec rm {} \;
else
    find $dir_name -type $type -name $name -exec rm -r {} \;
fi