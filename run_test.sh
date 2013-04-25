#!/bin/bash
cd test && make && cd ..
./colorcuda-memcheck ./test/check_bug
