User Guide
==========

Configuration
-------------

Guides and documentation of how to configure different settings and install special packages

Accessing course material
-------------------------

If the course material is downloaded either directly or using `git`, we can use (assuming that the code is in the specific folder)

.. code-block:: python

    import sys
    sys.path.insert(0,'C:\\code\\python_for_the_financial_economist\\')

Note that in `colab`, we need to install Python 3.8

.. code-block::

    !wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh
    !chmod +x mini.sh
    !bash ./mini.sh -b -f -p /usr/local
    !conda install -q -y jupyter
    !conda install -q -y google-colab -c conda-forge
    !python -m ipykernel install --name "py38" --user

If we are using `colab`, we can get access to code using

.. code-block::

    import os
    import sys

    user = "staxmetrics"
    repo = "python_for_the_financial_economist"

    # remove local directory if it already exists
    if os.path.isdir(repo):
        !rm -rf {repo}

    !git clone https://github.com/{user}/{repo}.git

    path = f"{repo}"
    if not path in sys.path:
        sys.path.insert(0, path)

    path = f"{repo}\data"
    if not path in sys.path:
        sys.path.insert(0, path)

It is possible to check which paths that are included using

.. code-block:: python

    print("\n".join(["'" + path + "'" for path in sys.path]))

If we want some data from the data folder, we can use

.. code-block:: python

    pd.read_csv('/content/python_for_the_financial_economist/data/feds200533.csv',
            skiprows=9, index_col=0, parse_dates=True)
