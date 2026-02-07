#!/usr/bin/env bash
# vim: set fileencoding=utf-8

###########################################################
# Author      : Kuan-Hsien Wu
# Contact     : jordankhwu@gmail.com
# Datetime    : 2026-02-07 17:21:34
# Description :
###########################################################

ssh -L 11434:localhost:11434 pop-os -t 'nvtop'
#ssh -L 11434:localhost:11434 pop-os-lan -t 'nvtop'
