# Contributing

Contributions are welcome and are greatly appreciated! Every little bit helps.
Contributions include reporting/fixing bugs, proposing/implementing features
(see our [Issue Tracker](https://github.com/qdglue/qdglue/issues)), writing
documentation in the codebase, and submitting feedback.

## Developing qdglue

Ready to contribute? Here's how to set up qdglue for local development.

1. [Fork](https://github.com/qdglue/qdglue/fork) the qdglue repo on GitHub.
1. Clone the fork locally:

   ```bash
   # With SSH:
   git clone git@github.com:USERNAME/qdglue.git

   # Without SSH:
   git clone https://github.com/USERNAME/qdglue.git
   ```

1. Create a branch for local development:

   ```bash
   git checkout -b name-of-bugfix-or-feature
   ```

1. Install the local copy and dev requirements into a virtual environment. For
   instance, with Conda, the following creates an environment at `./env`.

   ```bash
   cd qdglue
   conda create --prefix ./env python=3.7 # 3.7 is the minimum version qdglue supports.
   conda activate ./env
   pip install -e .[dev]
   ```

1. We roughly follow the
   [Google Style Guide](https://google.github.io/styleguide/pyguide.html) in our
   codebase by using black, isort, and pylint to enforce code format and style.
   To automatically check for formatting and style every time you commit, we use
   [pre-commit](https://pre-commit.com). Pre-commit should have already been
   installed with `.[dev]` above. To set it up, run:

   ```bash
   pre-commit install
   ```

1. Now make the appropriate changes locally. If relevant, make sure to write
   tests for your code in the `tests/` folder.

1. Auto-format and lint your code using black, isort, and pylint. We highly
   recommend installing editor plugins that perform these operations on save,
   but you can also run on the command line:

   ```bash
   black FILES
   isort FILES
   pylint FILES
   ```

   Note that these checks are also run by pre-commit whenever you commit your
   code.

1. After making changes, check that the changes pass the tests:

   ```bash
   pytest tests/
   make test # ^ same as above
   ```

   And to run the benchmarks:

   ```bash
   pytest -c pytest_benchmark.ini
   make benchmark # ^ same as above
   ```

1. Add your change to the changelog for the current version in `HISTORY.md`.

1. Commit the changes and push the branch to GitHub:

   ```bash
   git add .
   git commit -m "Detailed description of changes."
   git push origin name-of-bugfix-or-feature
   ```

1. Submit a pull request through the GitHub web interface.

## Instructions

### Running a Subset of Tests

To run a subset of tests, use `pytest` with the directory name, such as:

```bash
pytest tests/core/archives
```

### Documentation

Documentation is primarily written in Markdown, as we use mkdocs.

To preview documentation, use:

```bash
make servedocs
```

This will show a localhost URL that you can open in your browser; any changes
will automatically reload the page.

### Adding a Tutorial

> (Forthcoming)

Tutorials are created in Jupyter notebooks that are stored under `tutorials/` in
the repo. To create a tutorial:

1. Write the notebook and save it under `tutorials/`.
1. Use cell magic (e.g. `%pip install qdglue`) to install qdglue and other
   dependencies.
   - Installation cells tend to produce a lot of output. Make sure to clear this
     output in Jupyter lab so that it does not clutter the documentation.
1. Before the main loop of the QD algorithm, include a line like
   `total_itrs = 500` (any other integer will work). This line will be replaced
   during testing (see `tests/tutorials.sh`) in order to test that the notebook
   runs end-to-end. By default, the tests run the notebook with
   `total_itrs = 5`. If this tutorial needs more (or less), modify the
   switch-case statement in `tests/tutorials.sh`.
1. Make sure that the only level 1 heading (e.g. `# Awesome Tutorial`) is the
   title at the top of the notebook. Subsequent titles should be level 2 (e.g.
   `## Level 2 Heading`) or higher.
1. If linking to the qdglue documentation, make sure to link to pages in the
   `latest` version on ReadTheDocs, i.e. your links should start with
   `https://docs.pyribs.org/en/latest/`

   > TODO: update link above

1. Add an entry into the toctree in `docs/tutorials.md` and add it to one of the
   lists of tutorials.
1. Check that the tutorial shows up on the Tutorials page when serving the docs.

### Adding an Example

> (Forthcoming)

Examples are created in Python scripts stored under `examples/` in the repo, and
their source is shown in the docs. To create an example:

1. Write the Python script and save it under `examples/`.
1. Add any dependencies at the top of the script with a `pip install` command
   (see existing examples for a sample of how to do this).

   > TODO(btjanaka): We currently install requirements in
   > examples/requirements.txt; may want to change this in the future.

1. Add a shell command to `tests/examples.sh` that calls the script with
   parameters that will make it run as quickly as possible. This helps us ensure
   that the script has basic correctness. Also call the `install_deps` function
   on the script file before running the script.
1. Add a Markdown file in the `docs/examples` directory with the same name as
   the Python file -- if the example is `examples/foobar.py`, the Markdown file
   will be `docs/examples/foobar.md`.
1. Add a title to the Markdown file, such as:
   ```
   # My Awesome Example
   ```
1. In the markdown file, include the following so that the source code of the
   example is displayed.
   ````
   ```{eval-rst}
   .. literalinclude:: ../../examples/EXAMPLE.py
       :language: python
       :linenos:
   ```
   ````
1. Add any other relevant info to the Markdown file.
1. Add an entry into the toctree and list of examples in `docs/examples.md`.
1. Check that the example shows up on the Examples page when serving the docs.

### Referencing Papers

When referencing papers, refer to them as `Lastname YEAR`, e.g. `Smith 2004`.
Also, prefer to link to the paper's website, rather than just the PDF.

### Deploying

(Forthcoming)

1. Create a PR into master after doing the following:
   1. Switch tutorial links from latest to stable with:
      ```bash
      make tutorial_links
      ```
      See [pyribs #300](https://github.com/icaros-usc/pyribs/pull/300) for why
      we do this.
   1. Update the version with `bump2version` by running the following for minor
      versions:
      ```bash
      bump2version minor
      ```
      or the following for patch versions:
      ```bash
      bump2version patch
      ```
   1. Add all necessary info on the version to `HISTORY.md`.
1. (Optional) Once the PR has passed CI/CD and been squashed-and-merged into
   master, check out the squash commit and locally run `make release-test`. This
   uploads the code to TestPyPI to check that the deployment works. If this
   fails, make fixes as appropriate.
1. Once the PR in step 1 and any changes in step 2 have passed CI/CD and been
   squashed-and-merged into master, locally tag the master branch with a tag
   like `v0.2.1`, e.g.
   ```bash
   git tag v0.2.1 HEAD
   ```
1. Now push the tag with
   ```bash
   git push --tags
   ```
1. Check that the version was deployed to PyPI. If it failed, delete the tag,
   make appropriate fixes, and repeat steps 2 and 3.
1. Write up the release on GitHub, and attach it to the tag.
1. Submit another PR which reverts the changes to the tutorial links.
   Specifically, while on master, make sure your workspace is clean, then revert
   the changes with:
   ```bash
   git checkout HEAD~ tutorials/
   ```
   And commit the result.

Our deployment process may change in the future as qdglue becomes more complex.
