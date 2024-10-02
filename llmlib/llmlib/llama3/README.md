
Installation for the quantized model in `llama_cpp`:
```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/usr/local/cuda-12.5 -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.5/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```