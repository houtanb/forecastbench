#!/bin/sh
set -e
pwd
ls -la
bundle exec jekyll build --destination "$MOUNTED_BUCKET"
