#!/bin/bash
# colorcuda-memcheck

# Uninstall:

prefix="$HOME/bin"
rm "$prefix/cuda-memcheck" "$prefix/colorcolorcuda-memcheck" "$HOME/.colorcuda-memcheckrc"
sed -i -e "s@^export PATH=\"$prefix:\$PATH\" #added by colorcuda-memcheck install\.sh\$@@" ~/.bashrc
rmdir "$prefix"
