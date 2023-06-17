# Conditional Musical Accompaniment Generation
AI 535 Deep Learning Final Project

## Results Webpage
https://sites.google.com/oregonstate.edu/ai535final

## HPC
Example srun bash command
```srun -p cascades -A cascades --gres=gpu:2 --mem=100G --pty bash```

## Enviroment
To avoid version control issues, use the following commands to load the python enviroment for this project:
```bash
python3 -m venv env
source env/bin/activate
module load python/3.10 cuda/11.7
pip3 install -r requirements.txt
```
If you add pakages please run before committing:
```bash
pip3 freeze > requirements.txt
```

## Files
### data.py
The data class called `TrackDataset` to initialize the dataset and sample data.

#### Attributes and Methods:
- `train_dataset = TrackDataset(data_path)` to initialize the dataset.

- `self.track_list` will return audio directory list:

    `[{'track': 'minibabyslakh/train/Track00001', 'bass': 'minibabyslakh/train/Track00001/bass/bass.wav', 'residuals': 'minibabyslakh/train/Track00001/bass/residuals.wav'}, ...]`

- `window_size` is the length of the sample in seconds, default in 10 seconds
  - use `self.set_window_size(window_size)` to change the window size if needed
- `sample_rate` the **needed** sample rate of the audio, the data loader will automatically resample the audio to this sample rate, default is 24000
  - use `self.set_sample_rate(sample_rate)` to change the sample rate if needed
- `self.__getitem__` the magic method to customize dataloader, it will automaticly resample the audio and random sample the audio.

### slakh-utils
For easier version control, the `slakh-utils` is a submodule from this [repo](https://github.com/shawn120/slakh-utils/tree/4118ea16222d11d295496845e898cd497c7b7673). To update this submodule:

```bash
git submodule init
git submodule update
```
read more about submodules:

[A submodule tutorial from Bitbucket](https://www.atlassian.com/git/tutorials/git-submodule)

[Submodule intro from official website](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

### archieve_scripts
The scripts used for some early data preprocessing:

`convert.sh` the bash script to convert audio from flac to wav using slakh-utils, if rerun is needed, move this file to upper directory.

`submix.sh` the bash script to generate submix, if rerun is needed, move this file to upper directory.

`check_submix.py` to check if the all submix is generated correctly

`split.py` to split audio into small chunks (abandoned methond, we are using sampling instead, so only save it for archieve purpose, never rerun it)

## Data Organization
### Subset convention:
Slakh2100-orig: original split.

Slakh2100-split2: move files to make sure no duplicate songs in train and test set. Can be transferd by `splits_v2.json`

Slakh2100-redux: Slakh2100-redux: The subset we are using, it removed the duplicate songs. Can be transferd by `redux.json`

### Data structure:
```
- fail_submix -> all the audio file which fails to generate submix, so omit this
- omitted -> some duplicated audio from the dataset, so we omit them to prevent information leakage (also didn't generate submix for them)
- train
└─── Trackxxxxx
│    └─── mix.wav -> full mixed audio (not needed for this project)
│    └─── bass
│         └─── bass.wav -> target bass audio
│         └─── residuals.wav -> residuals audio
│    ... -> other files not used in this project
- validation
└─── Trackxxxxx
│    ...
- test
└─── Trackxxxxx
│    ...
```
