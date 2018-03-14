# quickdraw-chainer

Google [Quick, Draw!](https://quickdraw.withgoogle.com/) doodle recognition model implementation in [Chainer](https://github.com/chainer/chainer)

Open source for 11th Software Innovation Contest.

## Files

- `download.sh`, `filelist.txt`: fetch simplifed datasets from [Google Cloud Platform](https://console.cloud.google.com/storage/browser/quickdraw_dataset)

- `dataset.py`: Preprocess the datasets (selection and division)

- `model.py`: Model definition, supports CPU and GPU (switch by `gpu_id`)

- `train.py`: Main functions for data reading and training

- `PadIterator.py`: A special version of `SerialIterator` for padding the batch and pass its length into model

- `/demo`: see `/demo/README.md`

## Usage

Execute `download.sh` to get dataset (you may need to modify it to store them in a proper folder), then `dataset.py` to preprocess, `train.py` to start training.

Logs, visualization and snapshots will automatically save in `/result`.
