# Keras Lambda Injection PoC

This repository demonstrates a Proof of Concept (PoC) for injecting a Lambda function into a Keras model. The injected Lambda layer contains a custom `exec` function that is executed during model inference.

## Prerequisites

- Python 3.x
- TensorFlow
- Git

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/wearetyomsmnv/keras_lambda_rce.git
    cd keras_lambda_rce
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install tensorflow keras
    ```

## Injecting the Lambda Layer

1. Run the `inject_model.py` script to create and save a Keras model with an injected Lambda layer:
    ```bash
    python keras_inject.py
    ```
  with custom params as 
  exec
  pyrun
  system

and input layer (you can create test model via kerasgenerator.py

    You should see the following message in the console:
    ```
    Model injected and saved as 'test_model.h5'
    ```

## Explanation

- **inject_model.py**:
    - Creates a Sequential Keras model with several layers.
    - Injects a Lambda layer that executes the custom `exec` function with the code: `print('This model has been hijacked!')`.
    - Saves the model in HDF5 format as `test_model.h5`.

## Notes

- The model is saved using the HDF5 format, which is currently considered legacy in TensorFlow. It is recommended to use the new `.keras` format for production models, but for the purposes of this PoC, we continue using HDF5.
- If you wish to use the new format, simply modify the saving/loading code:
    ```python
    model.save('test_model.keras')
    ```

## License

This PoC is open source and is distributed under the MIT License.

PoC based on https://hiddenlayer.com/research/models-are-code/

## Questions 

t.me/wearetyomsmnv
