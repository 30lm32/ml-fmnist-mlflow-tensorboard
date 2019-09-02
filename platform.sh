#!/bin/sh

if [ $# -ne 1 ]
then
    echo "Usage: $0 <build|up|down>"
    exit
fi

COMMAND=$1

. ./platform_env.sh

case $COMMAND in
    build)
        docker-compose build;;
    up)
        docker-compose up;;
    down)
        docker-compose down
        docker-compose stop
        ;;
    *) echo "Usage: $0 <build|up|down>";;
esac




