---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

### Description

Please, describe briefly what the issue is

### Code example

If possible, write a minimum working example that reproduces the bug,
e.g:

```python
import madflow
madflow.broken_function()
```

### Additional information

Does the problem occur in CPU or GPU?
If GPU, how many? Which vendor? Which version of cuda or rocm are you using?

e.g:

```bash
nvcc --version
```

Please include the version of madflow and tensorflow that you are running. Running the following python script will produce useful information:

```python
import tensorflow as tf
import madflow

print(f"Madflow version: {madflow.__version__}")
print(f"Tensorflow: {tf.__version__}")
print(f"tf-mkl: {tf.python.framework.test_util.IsMklEnabled()}")
print(f"tf-cuda: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")
```
