#!/bin/sh

set -e

git config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
    while read path_key local_path
    do
        url_key=$(echo $path_key | sed 's/\.path/.url/')
        url=$(git config -f .gitmodules --get "$url_key")
        # if exist, continue
        if [ -d $local_path ]; then
            echo "Skip $local_path"
            continue
        fi
        git submodule add $url $local_path
    done