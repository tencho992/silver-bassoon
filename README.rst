XFrames 0.3 Library (BETA)
==========================

The XFrames Library provides a consistent and scalable data science
library that is built on top of industry-standard open source
technologies. XFrames provides the following advantages compared to other
DataFrame implementations:

-  A simple and well-tested Data Science library and Python based
   interface.
-  Powerful abstraction over underlying scalable big data and machine
   learning frameworks: Apache Spark, Spark DataFrames and ML libraries.
-  Dockerized container that bundles IPython notebooks, scientific
   libraries, Apache Spark and other dependencies for painless setup.
-  The library is extensible, allowing developers to add their own
   useful features and functionality.

How XFrames Benefits You
------------------------

If you're a data scientist, XFrames will isolate framework dependencies
and their configuration within a single disposable, containerized
environment, without compromising on any of the tools you're used to
working with (notebooks, dataframes, machine learning and big data
frameworks, etc.). Once you or someone else creates a single XFrames
container, you just need to run the container and everything is
installed and configured for you to work. Other members of your team
create their development environments from the same configuration, so
whether you're working on Linux, Mac OS X, or Windows, all your team
members are running data experiments in the same environment, against
the same dependencies, all configured the same way. Say goodbye to
painful setup times and "works on my machine" bugs.

Minimum Requirements
--------------------

Spark 1.6 or 2.1.

*Linux*:

-  Ubuntu 12.04 and above
-  Docker >= 1.5 installation

*Mac*:

-  Docker >= 1.5 installation

*Windows*

-  Run in VM

Getting Started
---------------

The easiest way to get started is to download the XFrames library, build a
Docker container that has everything you need, and run using an ipython notebook
within Docker.

Download Library
----------------

Clone XFrames this way::

    git clone https://github.com/chayden/xframes.git

Build Docker Container
----------------------

Go to the docker directory and follow the build instructions in
README.md.

Review Introductory Presentation
--------------------------------

After starting docker container, browse to http://localhost:7777/tree.
Then open info/Presentation.ipynb.  If you execute the cells in this
notebook, then XFrames is up and running.

Documentation
-------------

You can view local documentation with the Docker container on http://localhost:8000.

You can also view documentation at http://xframes.readthedocs.org/



Install Library On Existing Spark Installation
----------------------------------------------

If you have an existing Spark installation that you already use with
pySpark, then you can simply install the library to work with that.

You can install using pip or you can get the source and either:

1. Include the xframes directory in PYTHONPATH, or
2. Build an archive (see below) and install it on a different machine.

Install With Pip
----------------

You can also install with pip::

    pip install xframes

Using XFrames Directory
-----------------------

If you want to run using the source distribution, the most direct way
is to include its xframes directory in PYTHONPATH::

    export PYTHONPATH=<path to xframes>:$PYTHONPATH

Building the Library
--------------------

If you want to make a zip file that you can use to install XFrames on a
different machine, go to the source main directory and run::

  python setup.py sdist --formats=zip

This will create a file dist/xframes-<version>.zip This file can be copied to
the server where you want to install xframes.

Install by::

    unzip xframes.<version>.zip
    cd xframes.<version>
    python setup.py install


Running XFrames
---------------
XFrames uses pySpark, so you have to have Spark set up.

You might have an existing Spark installation running in Cluster Mode,
managed by the the Standalone, YARN, or Mesos cluster manager.
In this case, you need to set "master" to point to the cluster, using one
of the configuration methods described below.

Setting Up Spark
----------------

If you do not already have Spark, it is easy to set it up in local mode.

Download spark from http://spark.apache.org/downloads.html

Get the tar.gz, uncompress it, and put it in some convenient directory.
The path to py4j is dependent on the spark version: this one works with spark 1.2.
Then set::

    export SPARK_HOME=<spark distribution>
    export PYTHONPATH=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.4-src.zip

You can test by running this program::

    test.py:
    from xframes import XFrame
    print XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})

Run::

    $ python test.py

This should print::

    +----+-----+
    | id | val |
    +----+-----+
    | 1  |  a  |
    | 2  |  b  |
    | 3  |  c  |
    +----+-----+
    [? rows x 2 columns]


You may notice that a great deal of debug output appears on stdout.
This is because, by default, Spark displays log output on stdout.
You can change this by supplying a log4j.properties file and setting
SPARK_CONF_DIR to the directory containing it.  There is a sample
config dir "conf" under the xframes install directory.  You can copy this
to your current directory and set::

    export SPARK_CONF_DIR=`pwd`/conf

Then when you run, you will see only the output that your program prints.

Running in a IPython Notebook
-----------------------------

XFrames works especially well in an IPython notebook.
If you set up Spark as outline above, by setting PYTHONPATH, SPARK_HOME
and SPARK_CONF_DIR before you launch the notebook server, then
you can run the same test program and get the expected results.

See the blog http://blog.cloudera.com/blog/2014/08/how-to-use-ipython-notebook-with-apache-spark/
for more information on how to set up an existing Spark installation to use with
iIPython notebook.


Running in a Virtual Environment
--------------------------------

XFrames works well in a virtual environment.

Create a virtual environment::

    virtualenv venv

And then install into it::

    source venv/bin/activate
    pip install xframes

XFrames includes support for pandas and matplotlib, which you can
install if you want to use them.  For exammple::

    pip install pandas
    pip install matplotlib

After this, make sure Spark is set up.  For example::

    export SPARK_HOME=~/tools/spark
    export PYTHONPATH=${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-0.10.4-src.zip

Then test::

    cat <<EOF test.py
    from xframes import XFrame
    print XFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
    EOF
    python test.py

If running in a notebook, you could run the notebook server and run the test program::

  ipython notebook


Configurating Spark
-------------------

Spark has a large number of configuration parameters, described at:
http://spark.apache.org/docs/latest/configuration.html

There are a number of ways to supply these configuration parameters.
One of these is to supply a file spark-defaults.conf, in the directory pointed
to by SPARK_CONF_DIR described above.  There is a template to guide you.
This works when you start a local spark instance.

To affect only the spark context used by a single XFrames program, you can
either provide XFrames-specific defaults, application-speficic configuration,
or you can supply configurations at run time.

For XFrames-specific defaults, edit the file "defaults.ini" found in the xframes
directory in the xframe installation.

For application-specific defaults, use a file "config.ini" in the current directory where you run
your XFrames application.  It is structured similarly to "defaults.ini".

To provide run-time configuration, use XFrame.init_context to set configuration parameters before
running any Spark operations.

License
-------

This SDK is provided under the 3-clause BSD `license <LICENSE>`__.
