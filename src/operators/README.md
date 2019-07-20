# Approximate static condensation method

Here we extend the [ASC](https://authors.elsevier.com/c/1ZIL6508HiGTG) method to 3D.

## Building

* [Install Amanzi](https://github.com/56th/amanzi/blob/56th/ASC/INSTALL),
* Clone [Singleton-Logger](https://github.com/56th/Singleton-Logger),
* Modify [CMakeLists.txt](https://github.com/56th/amanzi/blob/56th/ASC/src/operators/CMakeLists.txt): change `<path>` in `set(LOGGER_PATH <path>)` to your path to a directory where you have `SingletonLogger.cpp` and `SingletonLogger.hpp`,
* In the directory where you built Amanzi, switch to `src/operators`. Run `make operators_diffusion -j`.

## Running

* Run `./operators_diffusion` and follow the instructions,
* The output will be put into `test/io`,
* `./operators_diffusion` generates `stdin.txt` based on your input. You can change parameters in `stdin.txt` and run `./operators_diffusion < stdin.txt`.
