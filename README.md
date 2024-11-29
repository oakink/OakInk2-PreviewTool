# OakInk2 Toolkit & Preview-Tool

## Setup dataset files.

    Download tarballs from [huggingface](https://huggingface.co/datasets/kelvin34501/OakInk-v2).
    You will need the data tarball and the preview version annotation tarball for at least one sequence, the object_raw tarball, the object_repair tarball and the program tarball.
    Organize these files as follow:
    ```
    data
    |-- data
    |   `-- scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS
    |-- anno_preview
    |   `-- scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.pkl
    |-- object_raw
    |-- object_repair
    `-- program
    ```

## OakInk2 Toolkit

1. Install the package.

    ```bash
    pip install .
    ```

    Optionally, install it with editable flags:
    ```bash
    pip install -e .
    ```

2. Check the installation.

    ```bash
    python -c 'from oakink2_toolkit.dataset import OakInk2__Dataset'
    ```

    It the command runs without error, the installation is successful.


## OakInk2 Preview-Tool

![oakink2_preview_tool](./doc/oakink2_preview_tool.gif)

1. Setup the enviroment.

    1. Create a virtual env of python 3.10. This can be done by either `conda` or python package `venv`.
    
        1. `conda` approach
            
            ```bash
            conda create -p ./.conda python=3.10
            conda activate ./.conda
            ```

        2. `venv` approach
            First use `pyenv` or other tools to install a python intepreter of version 3.10. Here 3.10.14 is used as example:

            ```bash
            pyenv install 3.10.14
            pyenv shell 3.10.14
            ```

            Then create a virtual environment:

            ```bash
            python -m venv .venv --prompt oakink2_preview
            . .venv/bin/activate
            ```
    
    2. Install the dependencies.

        Make sure all bundled dependencies are there.
        ```bash
        git submodule update --init --recursive --progress
        ```

        Use `pip` to install the packages:
        ```bash
        pip install -r requirements.txt
        ```

2. Download the [SMPL-X model](https://smpl-x.is.tue.mpg.de/download.php)(version v1.1) and place the files at `asset/smplx_v1_1`.
    
    The directory structure should be like:
    ```
    asset
    `-- smplx_v1_1
       `-- models
            |-- SMPLX_NEUTRAL.npz
            `-- SMPLX_NEUTRAL.pkl
    ```

3. Launch the preview tool:
    ```bash
    python -m launch.viz.gui --cfg config/gui__preview.yml
    ```

    Or use the shortcut:
    ```bash
    oakink2_viz_gui --cfg config/gui_preview.yml
    ```

4. (Optional) Preview task in segments.

    1. Download the [MANO model](https://mano.is.tue.mpg.de)(version v1.2) and place the files at `asset/mano_v1_2`.

        The directory structure should be like:
        ```
        asset
        `-- mano_v1_2
            `-- models
                |-- MANO_LEFT.pkl
                `-- MANO_RIGHT.pkl
        ```

    2. Launch the preview segment tool (press enter to proceed). Note `seq_key` should contain '/' rather than '++' as directory separator.

        ```bash
        python -m oakink2_preview.launch.viz.seg_3d --seq_key scene_0x__y00z/00000000000000000000__YYYY-mm-dd-HH-MM-SS
        ```

        Or use the shortcut:
        ```bash
        oakink2_viz_seg3d --seq_key scene_0x__y00z/00000000000000000000__YYYY-mm-dd-HH-MM-SS
        ```

5. (Optional) View the introductory video on [youtube](https://www.youtube.com/watch?v=Xtk07q5HiOg).


## Dataset Format

+ `data/scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS`
+ `anno/scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.pkl`

    This pickle stores a dictonary under the following format:
    ```
    {

    }
    ```
+ `object_{raw,scan}/obj_desc.json`
+ `object_{raw,scan}/align_ds`
+ `program/desc_info/scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.json`
+ `program/initial_condition_info/scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.json`
+ `program/pdg/scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.json`
+ `program/program_info/scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.json`