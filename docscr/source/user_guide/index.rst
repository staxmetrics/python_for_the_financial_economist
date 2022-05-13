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



If we are using `colab`, we can directly install relevant modules using the following

.. code-block::

    import os

    user = "staxmetrics"
    repo = "python_for_the_financial_economist"

    url = f"git+https://github.com/{user}/{repo}.git"
    !pip install --upgrade {url}

Alternatively, we can use

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