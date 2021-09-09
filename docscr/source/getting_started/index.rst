Getting started
===============

This note describes to get started with the course *Python for the Financial Economist* including installation of relevant software.

Best,
Johan


Installation of relevant software
---------------------------------

Python
^^^^^^

Navigate to `python download <https://www.python.org/downloads/>`_ and download and install relevant Python release. I have installed `Python 3.8`.

PyCharm
^^^^^^^^

We will use the `PyCharm` IDE in this course that can de downloaded `here <https://www.jetbrains.com/pycharm/download/#section=windows>`_. Select the `professional` version.

A free student license can be obtained `here <https://www.jetbrains.com/community/education/#students>`_.

The license can be registered by selecting `Help > Register` in the toolbar when opening `PyCharm`
(see `here <https://www.jetbrains.com/help/pycharm/register.html>`_).


Change some settings
""""""""""""""""""""

I will be using the `numpy` docstring format. Change it in `File > Settings > Tools > Python Integrated Tools`:

.. image:: change_docstring_format.jpg
    :scale: 70 %
    :align: center

Git and git bash
^^^^^^^^^^^^^^^^

Navigate to `git download <https://git-scm.com/downloads>`_ and download and install the newest release.

Github repository
-----------------

Most of the relevant course material can be found in the `github repository <https://github.com/staxmetrics/python_for_the_financial_economist>`_.

To clone the github repository (the master branch), open Git Bash and navigate to the folder where you want the local branch.
I use the folder `C:\\code` for repositories.

In Git Bash type

.. code-block::

    cd c:\code
    git clone https://github.com/staxmetrics/python_for_the_financial_economist.git


Alternatively, it will also be possible directly in PyCharm. Navigate to VCS->Git->Clone.

Now you will have a local version of the github repository on your computer!

After you have cloned the github repository, you can `right click` on the folder and choose `Open Folder As PyCharm Project`.
If this not work, then you can open a specific project from PyCharm.

Configuration of virtual environment (optional)
-----------------------------------------------

It is possible to use the Python system interpreter.

However, it is recommended to be working with different `virtual environments <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_
for different projects. This allows us to have control over which specific packages and versions of these packages that we are using.

After you have opened the project, then we need to configure the virtual environment. Go to `File > Settings > Project: Python for ...` and
select `Python interpreter`. Press the symbol in the red circle (see below) and select `Add`

.. image:: virt_env_1.jpg
    :scale: 70 %
    :align: center


Select the folder where you want the virtual environment and the name you want to use. I have the folder `C:\\environments` for virtual environments.
You also need to select the location of python installation. Press `OK`.

.. image:: virt_env_2.jpg
    :scale: 60 %
    :align: center


When selecting the terminal you should be able to see that you are working with your particular virtual environment, e.g.

.. image:: virt_env_3.jpg
    :scale: 50 %
    :align: center

Install relevant packages
^^^^^^^^^^^^^^^^^^^^^^^^^

The `requirements.txt` file contains the majority of the python packages needed during the course. You can install all of them using (in the terminal)

.. code-block:: console

    pip install -r requirements.txt


Folder structure
----------------

The folder structure is presented below.

::

    python_for_the_financial_economist
    ├── codelib
    │   ├── dal
    │   └── statistics
    │   └── visualization
    ├── data
    ├── docs
    ├── examples_notebooks
    ├── lectures_and_exercises
    ├── tests
    └── requirements.txt


Jupyter Notebooks
-----------------

`Jupyter <https://jupyter.org/>`_ is a browser-based way of interacting with Python and is especially useful when working
and interacting with data and want to visualize and do calculations on the fly. It is generally not suited for developing
bigger applications.

We will be using Jupyter notebooks extensively during the course.

We can open Jupyter notebooks using the `cmd prompt` or the terminal in PyCharm by navigating to the relevant folder and
applying the command

.. code-block:: console

    jupyter notebook

If you run the command from the root of `python_for_the_financial_economist`, you should see something like

.. image:: jupyter_1.jpg
    :scale: 70 %
    :align: center

Access virtual environment in Jupyter notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To be able to access the virtual environment from a Jupyter notebook, we need run the following command in the terminal

.. code-block:: console

    python -m ipykernel install --user --name=name_of_venv

See e.g. `this blog <https://janakiev.com/blog/jupyter-virtual-envs/>`_ for further details.


Pulling newest update to local repository
----------------------------------------

I will continuously add new material to the github repository. To pull the newest version, you need to download it to your computer.
This can be done directly from PyCharm by navigating to VCS->Git->Pull (on a Windows machine).

Working with your own code and notebooks
----------------------------------------

It should be noted that if you just start working on the master branch in the repository, then you will likely get `merge conflicts`
when trying to pull changes from the github (remote) repository.

There are several possibilities to work with your own code including

New branch in the repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may make a new branch in the repository `python_for_the_financial_economist`. You will then have to merge changes to the master branch into your own branch
to have the newest material.

New repository
^^^^^^^^^^^^^^

You can create your own repository.

A folder with code and notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest approach will be to simple have a folder somewhere on your computer with notebooks and scripts.