# OakInk2-Preview-Tool

1. Setup dataset files.

    Download tarballs from [huggingface](https://huggingface.co/datasets/kelvin34501/OakInk-v2).
    You will need the data tarball and the preview version annotation tarball for at least one sequence, the object_preview tarball and the program tarball.
    Organize these files as follow:
    ```
    data
    |-- data
    |   `-- scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS
    |-- anno_preview
    |   `-- scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.pkl
    |-- object_preview
    `-- program
    ```

2. Setup the enviroment.

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
            python -m venv .venv --prompt mocap_blender
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

3. Download the [SMPL-X model](https://smpl-x.is.tue.mpg.de/download.php)(version v1.1) and place the files at `asset/smplx_v1_1`.
    
    The directory structure should be like:
    ```
    asset
    `-- smplx_v1_1
       `-- models
            |-- SMPLX_NEUTRAL.npz
            `-- SMPLX_NEUTRAL.pkl
    ```


4. Launch the preview tool:
    ```bash
    python -m launch.viz.gui --cfg config/gui__preview.yml
    ```

5. (Optional) View the introductory video on [youtube](https://www.youtube.com/watch?v=Xtk07q5HiOg).
