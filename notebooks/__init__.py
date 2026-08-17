"""Analysis notebooks and their presentation helpers.

This package is **not** part of the deployed product. It exists so
``notebooks/analysis_support.py`` can be imported as ``notebooks.analysis_support`` rather
than resolving through an implicit namespace package, which makes the import explicit to a
reader and to tooling.

Why the boundary matters
------------------------
``src/`` is the deployed package: ``requirements-serve.txt`` installs scikit-learn, pandas
and FastAPI, and nothing else. The helpers here depend on matplotlib, seaborn and
``IPython.display``, none of which reach the serving image. Keeping them out of ``src/``
means the API cannot import a dependency the container does not have - a failure that would
otherwise appear only at runtime, in production.

``.gcloudignore`` excludes this directory from the Cloud Run upload for the same reason.
"""
