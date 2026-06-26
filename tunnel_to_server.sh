#!/usr/bin/env bash
# vim: set fileencoding=utf-8

###########################################################
# Author      : Kuan-Hsien Wu
# Contact     : jordankhwu@gmail.com
# Datetime    : 2026-02-07 17:21:34
# Description :
###########################################################

#ssh -L 11434:localhost:11433 ubuntu@ubuntu-desktop -t 'nvtop'

ssh \
    -L 11434:localhost:11434 \
    -L 8001:localhost:8001 \
    -L 8002:localhost:8002 \
    -L 8080:localhost:8080 \
    -L 8082:localhost:8082 \
    pop-os -t 'tmux a -t admin'
