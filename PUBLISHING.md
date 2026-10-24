# Publishing `nest-st` 0.2.1

PyPI release numbers cannot be reused. Version `0.2.0` is already published,
so this source tree uses `0.2.1` in both `pyproject.toml` and
`src/nest_st/__init__.py`.

The commands below are written for **Anaconda Prompt / Windows CMD**, not
PowerShell. The example source directory is:

```text
F:\NEST\raw_code\nest_st_package
```

## 1. Create a clean release environment

No pre-existing `nest_clean` environment is required:

```bat
conda create -n nest_release python=3.11 -y
conda activate nest_release
cd /d F:\NEST\raw_code\nest_st_package
python -m pip install --upgrade pip build twine
```

Building and checking distributions do not import the scientific runtime, so
PyTorch and Scanpy are not required in this release-only environment.

## 2. Remove old generated artifacts and build

```bat
cd /d F:\NEST\raw_code\nest_st_package
if exist build rmdir /s /q build
if exist build_check rmdir /s /q build_check
if exist dist rmdir /s /q dist
if exist src\nest_st.egg-info rmdir /s /q src\nest_st.egg-info

python -m compileall -q src\nest_st
python -m build
python -m twine check dist\*
dir dist
```

Expected files:

```text
dist\nest_st-0.2.1-py3-none-any.whl
dist\nest_st-0.2.1.tar.gz
```

## 3. Optional TestPyPI rehearsal

TestPyPI and PyPI use different accounts and tokens. Specifying the upload URL
directly avoids requiring a `testpypi` section in `%USERPROFILE%\.pypirc`:

```bat
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist\*
```

Use `__token__` as the username and the complete TestPyPI token as the
password. Check the uploaded artifact without resolving scientific dependencies:

```bat
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps nest-st==0.2.1
python -c "import importlib.metadata as m; print(m.version('nest-st'))"
```

## 4. Publish to PyPI manually

If GitHub Trusted Publishing is not configured, upload with a project-scoped
PyPI token:

```bat
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist\*
```

Use `__token__` as the username and the complete PyPI token as the password.
Never store a token in this repository.

Verify the release in a new runtime environment:

```bat
conda create -n nest_verify python=3.11 -y
conda activate nest_verify
python -m pip install nest-st==0.2.1
python -c "import nest_st; print(nest_st.__version__)"
nest-st --help
```

## 5. Copy the release source into a GitHub checkout

Do not run `git init` inside `nest_st_package` when the remote repository
already exists. Clone a clean checkout beside it:

```bat
cd /d F:\NEST\raw_code
git clone https://github.com/lty10086/NEST.git NEST_GitHub
```

If `NEST_GitHub` already exists, update it instead:

```bat
cd /d F:\NEST\raw_code\NEST_GitHub
git switch main
git pull origin main
```

Copy only maintained source files; this preserves remote-only folders such as
`Databases`:

```bat
xcopy /E /I /Y "F:\NEST\raw_code\nest_st_package\src\nest_st" "F:\NEST\raw_code\NEST_GitHub\src\nest_st"
xcopy /E /I /Y "F:\NEST\raw_code\nest_st_package\tests" "F:\NEST\raw_code\NEST_GitHub\tests"
if not exist "F:\NEST\raw_code\NEST_GitHub\.github\workflows" mkdir "F:\NEST\raw_code\NEST_GitHub\.github\workflows"
copy /Y "F:\NEST\raw_code\nest_st_package\.github\workflows\publish.yml" "F:\NEST\raw_code\NEST_GitHub\.github\workflows\publish.yml"
copy /Y "F:\NEST\raw_code\nest_st_package\pyproject.toml" "F:\NEST\raw_code\NEST_GitHub\pyproject.toml"
copy /Y "F:\NEST\raw_code\nest_st_package\README.md" "F:\NEST\raw_code\NEST_GitHub\README.md"
copy /Y "F:\NEST\raw_code\nest_st_package\PUBLISHING.md" "F:\NEST\raw_code\NEST_GitHub\PUBLISHING.md"
copy /Y "F:\NEST\raw_code\nest_st_package\LICENSE" "F:\NEST\raw_code\NEST_GitHub\LICENSE"
copy /Y "F:\NEST\raw_code\nest_st_package\.gitignore" "F:\NEST\raw_code\NEST_GitHub\.gitignore"

cd /d F:\NEST\raw_code\NEST_GitHub
git status
git diff --check
git add src tests .github pyproject.toml README.md PUBLISHING.md LICENSE .gitignore
git commit -m "Release nest-st 0.2.1"
git push origin main
git tag -a v0.2.1 -m "nest-st 0.2.1"
git push origin v0.2.1
```

## 6. Recommended GitHub Trusted Publishing path

The workflow `.github/workflows/publish.yml` builds and uploads when a GitHub
Release is published. Configure PyPI once:

1. Open **PyPI > nest-st > Manage > Publishing**.
2. Add owner `lty10086`, repository `NEST`, workflow `publish.yml`, and
   environment `pypi`.
3. Create the `pypi` environment in the GitHub repository settings.
4. Push the source and tag `v0.2.1` as shown above.
5. Publish a GitHub Release for `v0.2.1`.

Choose either Trusted Publishing or the manual Twine upload for a release; do
not run both, because PyPI rejects duplicate version files.
