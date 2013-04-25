#!/bin/bash
# colorcuda-memcheck

# Quick Install:

prefix="$HOME/bin"
[ ! -d "$prefix" ] && mkdir "$prefix"
cp colorcuda-memcheck "$prefix"
sed -e "/^nvcc:/s@/opt/cuda/bin/cuda-memcheck@$(which cuda-memcheck)@" \
  example.colorcuda-memcheckrc > ~/.colorcuda-memcheckrc
which colorcuda-memcheck 2>&1 >/dev/null || \
  echo "export PATH=\"$prefix:\$PATH\" #added by colorcuda-memcheck install.sh" >> ~/.bashrc
source ~/.bashrc
ln -s colorcuda-memcheck "$prefix/cuda-memcheck"

# Make sure we can execute colornvcc
if  ! which colorcuda-memcheck 2>&1 >/dev/null
then
    echo "Error installing colorcuda-memcheck: can't run colorcuda-memcheck"
fi

# Make sure nvcc will execute "$prefix/nvcc"
if [ "$(which cuda-memcheck)" != "$prefix/cuda-memcheck" ]
then
    echo "Error installing colorcuda-memcheck: cuda-memcheck points to wrong executable"
fi

echo "Finished installing colorcuda-memcheck"

