# Quick, Draw! Demo

A HTML-based demo for [Quick, Draw!](https://quickdraw.withgoogle.com/).

![image](/demo/preview.png)

## Files

- `drawing.html`: A simple drawing board implemented in JavaScript, supports mouse, touching and pen

- `get_predict.py`: Simplify and prediction

- `predictor.py`: Model for prediction (a little different from `../model.py`, removing mask function for batch_size=1)

- `model.npz`, `training.json.classes`: pre-trained model (include 20 types (see `training.json.classes`) of doodles)

## Usage

You may use a simple HTTP server (such as [Flask](https://github.com/pallets/flask)) to host requests from `drawing.html`, parse json body into a dict, and pass it to `get_predict()`, it will returns `topN` results.
