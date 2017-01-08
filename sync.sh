#!/bin/bash 
if [ -z "$1" ]
then echo $1
else
    rsync -rva . kpi:labs --progress
fi
