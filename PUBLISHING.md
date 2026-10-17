# Publishing `nest-st`

The distribution name `nest-st` already has release `0.1.0` on PyPI, so this
package is versioned as `0.2.0`. PyPI does not allow a released version number
to be reused.

## 1. Work in a clean environment

From this directory:

```powershell
py -m venv .venv-build
.\.venv-build\Scripts\Activate.ps1
python -m pip install --upgrade pip build twine
```

The build environment does not need PyTorch or the scientific runtime because
building reads metadata without importing `nest_st`.

## 2. Build and inspect the distributions

Remove artifacts from an older local build, then create a source archive and a
wheel:

```powershell
Remove-Item -Recurse -Force .\dist -ErrorAction SilentlyContinue
python -m build
python -m twine check .\dist\*
```

The expected files are:

```text
dist/nest_st-0.2.0-py3-none-any.whl
dist/nest_st-0.2.0.tar.gz
```

Before a real release, install the wheel in the same type of CPU or CUDA
environment used for NEST and run a small `missing_gene` and `dropout` example.

## 3. Optional TestPyPI rehearsal

TestPyPI and PyPI use separate accounts and tokens. Upload with:

```powershell
python -m twine upload --repository testpypi .\dist\*
```

When prompted, use `__token__` as the username and paste the complete TestPyPI
token (including its `pypi-` prefix) as the password. Because TestPyPI does not
mirror every scientific dependency, validate the uploaded artifact without
dependency resolution:

```powershell
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps nest-st==0.2.0
```

## 4. Publish to PyPI

Create a project-scoped token for `nest-st` in the PyPI project settings, then:

```powershell
python -m twine upload .\dist\*
```

Use `__token__` as the username and the full PyPI token as the password. Do not
commit a token or store it in this project.

## 5. Verify the public release

In a fresh runtime environment:

```powershell
python -m pip install --upgrade nest-st==0.2.0
python -c "import nest_st; print(nest_st.__version__)"
nest-st --help
```

For the next release, update the version in both `pyproject.toml` and
`src/nest_st/__init__.py`, rebuild from an empty `dist` directory, and upload the
new version.

## Recommended automated release

The included `.github/workflows/publish.yml` supports PyPI Trusted Publishing.
After pushing the package to GitHub:

1. In PyPI, open **nest-st > Manage > Publishing**.
2. Add a GitHub publisher for owner `lty10086`, repository `NEST`, workflow
   `publish.yml`, and environment `pypi`.
3. In GitHub, create the `pypi` environment and optionally require approval.
4. Publish a GitHub Release; the workflow builds and uploads without a stored
   long-lived token.

