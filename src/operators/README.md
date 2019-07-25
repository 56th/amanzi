# Approximate static condensation method

Here we extend the [ASC](https://authors.elsevier.com/c/1ZIL6508HiGTG) method to 3D.

## Building

* Clone this repo of Amanzi: `$ git clone git@github.com:56th/amanzi.git`. The default branch shall be `56th/asc`.
* Clone [Singleton-Logger](https://github.com/56th/Singleton-Logger),
* Clone [Tangram](https://github.com/56th/tangram) with its submodules: `$ git clone --recursive git@github.com:56th/tangram.git`. The default branch shall be `56th/amanzi`. In your Tangram directory, go to `wonton`. The default branch for `wonton` shall be `56th/amanzi`. From `wonton`, go to `wonton/mesh/amanzi` and make sure that `amanzi_mesh_wrapper.h` exists,
* Modify [CMakeLists.txt](https://github.com/56th/amanzi/blob/56th/ASC/src/operators/CMakeLists.txt): change `<path>` in `set(TANGRAM_PATH <path>)` to your Tangram repository directory, and change `<path>` in `set(LOGGER_PATH <path>)` to your Singleton-Logger repository directory,
* [Install Amanzi](https://github.com/56th/amanzi/blob/56th/ASC/INSTALL),
* In the directory where you built Amanzi, switch to `src/operators`. Run `$ make operators_diffusion -j`.

## Running

* Run `$ ./operators_diffusion` and follow the instructions,
* The output will be put into `test/io`,
* `$ ./operators_diffusion` generates `stdin.txt` based on your input. You can change parameters in `stdin.txt` and run `$ ./operators_diffusion < stdin.txt`.
