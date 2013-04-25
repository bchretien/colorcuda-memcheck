Install
=======

You should already have setup `cuda-memcheck` in your path.

*Note*: if you cannot run `cuda-memcheck -h`, `cuda-memcheck` is not in your path.


Quick Install
-------------

Run the `./install.sh` script.

Manual Install
--------------

(e.g. installing into `~/bin`)

1. Copy `colorcuda-memcheck` into `~/bin`:

    $ cp colorcuda-memcheck ~/bin

2. Copy `example.colorcuda-memcheckrc` to `~/.colorcuda-memcheckrc`:

    $ cp example.colorcuda-memcheckrc ~/.colorcuda-memcheckrc

3. Find your current `cuda-memcheck` path:

    $ which cuda-memcheck

4. Edit `~/.colorcuda-memcheckrc` to set the `cuda-memcheck` path:

    $ nano ~/.colorcuda-memcheckrc

Edit the line which starts with **cuda-memcheck:** and use the
path you found in step 3, e.g.:

    cuda-memcheck: /opt/cuda/bin/cuda-memcheck

5. Create a symlink from `colorcuda-memchec` to `cuda-memcheck`:

    $ ln -s ~/bin/colorcuda-memcheck ~/bin/cuda-memcheck

6. Add `~/bin` to your `PATH`:

    $ export PATH="~/bin:$PATH"

7. Add this export line to your `~/.bashrc` so that your `PATH`
will be automatically set up in future shell sessions.

    $ echo 'export PATH="~/bin:$PATH"' >> ~/.bashrc

8. Make sure you are using the `~/bin/cuda-memcheck` executable

    $ which cuda-memcheck

This command should return **~/bin/cuda-memcheck**.

:)
