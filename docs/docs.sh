#!/bin/sh

# Copyright (C) 2020 The now-u Authors
# Maintainer: Maxime Robeyns <maximerobeyns@gmail.com>

cd /docs

case $1 in
    "")
        make html
        exit
        ;;
    "watch")
        make html
        live-server -p 8080 /docs/build/html &
        inotifywait -m /docs/source -e modify |
            while read path action file; do
                make html
            done
        exit
        ;;
    # TODO add pdf or texinfo outputs here if necessary
    *)
        echo "Error: unknown parameter provided: ${1}."
        exit 1
        ;;
esac
