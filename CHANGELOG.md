# Changelog

All notable changes to this project will be documented in this file.

## [1.3.2] - 2025-12-05

### Added
- **MLP Model Enhancement**: Added `get_params()` method to the MLP class for scikit-learn compatibility with model screening tools
- **GitHub Actions Workflow**: Created comprehensive CI/CD pipeline (`test.yml`) for cross-platform testing on Ubuntu, macOS, and Windows with native coverage reporting
- **Mixed Precision Workaround**: Added mixed precision policy initialization in LorentzLorenz to prevent `ml_dtypes.float4_e2m1fn` compatibility errors with TensorFlow 2.19+
- **Keras 3 Compatibility**: Updated Adam optimizer imports and instantiations to use Keras 3 standard (removed deprecated `.legacy` module and `decay` parameter)
- **Dictionary-based Metrics**: Implemented Keras 3-compliant multi-output model metrics using output name mapping to prevent duplicate metric naming errors
- **GitHub Native Coverage Reporting**: Added artifact uploads for coverage XML reports from each test run on different platforms
- **GitHub Actions Badge**: Added workflow status badge to README for visibility into test status

### Changed
- **Loss Specification**: Updated LorentzLorenz model compilation to use list of losses for each output instead of single loss string, matching Keras 3 requirements
- **Metrics Configuration**: Changed from list-based metrics to dictionary-based metrics for multi-output models to ensure unique metric names in Keras 3
- **AutoML Multi-core Support**: Enhanced `model_screener.py` with improved multi-core processing capabilities
- **Model Screener**: Updated `test_hyp()` compatibility for better parameter tracking and reporting
- **CI/CD Infrastructure**: Migrated from Travis CI to GitHub Actions with improved coverage reporting
- **Version Tracking**: Updated README to reference GitHub releases instead of PyPI for latest version
- **Version Number**: Bumped to 1.3.2 to reflect Keras 3 compatibility and infrastructure improvements

### Removed
- **Travis CI Configuration**: Removed outdated `.travis.yml` file (superseded by GitHub Actions)
- **Codecov Integration**: Removed external Codecov service dependency in favor of GitHub-native coverage artifact uploads
- **PyPI Badge**: Replaced with GitHub releases badge as repo is ahead of PyPI

### Fixed
- **Keras 3/TensorFlow 2.19 Compatibility**: Fixed "Found two metrics with the same name" error by implementing proper output naming and dictionary-based metrics
- **Adam Optimizer**: Removed incompatible `decay` parameter for Keras 3 Adam optimizer initialization
- **Mixed Precision Issues**: Resolved `ml_dtypes.float4_e2m1fn` AttributeError by setting global mixed precision policy to float32
- **PyTorch Installation**: Added OS-specific PyTorch installation in GitHub Actions workflow
- **OpenBabel Import**: Fixed openbabel import failures in GitHub Actions by adding openbabel-wheel pip installation alongside conda installation

### Technical Details

#### Commits Included:
1. **b83abd9** - patch to published models
   - Added `get_params()` method to MLP class
   - Fixed LorentzLorenz model metrics for Keras 3 compatibility
   - Updated notebook documentation

2. **dba73a6** - backwards compatibility fixes
   - Updated Adam optimizer imports (removed `.legacy`)
   - Fixed mixed precision initialization in LorentzLorenz
   - Updated test imports for consistency

3. **5739610** - AutoML multi-core update
   - Enhanced `model_screener.py` with improved parallelization
   - Updated `space.py` for better genetic algorithm integration
   - Modified MLP to support model screening via `get_params()`
   - Updated test cases for AutoML screening

4. **8bbe071** - Updated readme and setup for local install
   - Updated README installation instructions
   - Modified setup.py for Python 3.12 compatibility

### Dependencies Updated
- **TensorFlow/Keras**: Now compatible with Keras 3 and TensorFlow 2.19+
- **PyTorch**: Added proper CPU-only installation for CI environments
- **System Libraries**: Added openbabel-wheel for proper pip installation alongside conda openbabel

### Testing
- All tests pass on Ubuntu, macOS, and Windows with Python 3.12
- Cross-platform CI validation implemented via GitHub Actions
- Coverage reporting integrated with Codecov

### Migration Guide for Users

If you're upgrading from the previous version, note these breaking changes:

1. **Adam Optimizer Parameters**: The `decay` parameter is no longer supported in Adam. Use `learning_rate` scheduling instead.
2. **Keras 3 Models**: Multi-output models now require dictionary-based metrics configuration:
   ```python
   metrics_dict = {
       'output_name': ['metric1', 'metric2'],
       ...
   }
   model.compile(metrics=metrics_dict)
   ```
3. **Mixed Precision**: Mixed precision is now disabled by default to ensure compatibility. Enable it explicitly if needed.

---

For more information on each change, see the individual commit messages or the pull request discussions.
