
# Keras Custom Training Loop on MNIST

This project demonstrates how to train a neural network in Keras/TensorFlow using a **custom training loop** instead of relying only on `model.fit()`.

The notebook is organized into three main parts. Each part builds on the previous one by adding a new training feature: first a manual loop, then accuracy tracking, and then a custom callback. At the end, the notebook also includes a standard Functional API classifier for comparison.

## Overview

The project uses the MNIST handwritten digit dataset and shows how low-level training works in TensorFlow:

1. load and preprocess MNIST
2. build a simple neural network classifier
3. define loss and optimizer manually
4. train the model batch by batch with `tf.GradientTape`
5. extend the loop by adding an accuracy metric
6. extend it again by adding a custom callback
7. compare this custom workflow with a standard compiled Keras model

## Part 1: Basic Custom Training Loop

introduce model

define loss function and optimizer

custom training loop

### What this part does

The first part builds a simple classifier and trains it manually without `model.fit()`.

### Model structure

- `Flatten(input_shape=(28, 28))`
- `Dense(128, activation='relu')`
- `Dense(10)`

This is a basic feedforward classifier:
- the `Flatten` layer turns each `28 x 28` image into a 784-dimensional vector
- the hidden layer learns intermediate features
- the final layer outputs 10 logits, one for each digit class

### What is added in Part 1

This part introduces the core custom loop pieces:

- `SparseCategoricalCrossentropy(from_logits=True)`
- `Adam` optimizer
- `tf.data.Dataset(...).batch(32)`
- `tf.GradientTape()` for gradient computation
- manual gradient updates with `optimizer.apply_gradients(...)`

This is the foundation of the notebook because it shows exactly how forward pass, loss computation, backpropagation, and parameter updates happen.

## Part 2: Custom Training Loop with Accuracy Metric

### Part two: adding an accuracy metric to monitor model performance

Define the model

defining loss function, optimization, and metric

creating custom training loop

### What this part does

The second part keeps the same classifier structure but improves the training loop by tracking model accuracy during training.

### Model structure

- `Flatten(input_shape=(28, 28))`
- `Dense(128, activation='relu')`
- `Dense(10)`

So the architecture is the same as Part 1. The main change is not the network itself, but the training logic.

### What is added in Part 2

This part adds:

- `SparseCategoricalAccuracy()` metric
- metric updates inside the custom loop
- printed monitoring for both loss and accuracy

This makes the loop more informative because now training is monitored not only by loss, but also by classification performance.

## Part 3: Custom Training Loop with Custom Callback

### Part three: creating custom callback to log additional metrics and information during training

define model

Loss Function, Optimizer, and Metric

custom training loop with custom callback

### What this part does

The third part again uses the same classifier, but now adds a custom callback class to log information at the end of each epoch.

### Model structure

- `Flatten(input_shape=(28, 28))`
- `Dense(128, activation='relu')`
- `Dense(10)`

### What is added in Part 3

This part adds:

- a custom callback class inheriting from `Callback`
- `on_epoch_end()` logging
- manual callback use alongside the custom loop

The callback prints the epoch number, loss, and accuracy at the end of each epoch. This shows how custom monitoring behavior can be added beyond the default Keras logging tools.

## Additional Section: Standard Functional API Model

adding hidden layers

output layer

create model

compile the model

train the model

evaluate the model

After the three main custom-loop parts, the notebook also includes a standard Keras model built with the Functional API.

### Model structure

- `Input(shape=(28, 28))`
- `Flatten()`
- `Dense(64, activation='relu')`
- `Dense(64, activation='relu')`
- `Dense(10, activation='softmax')`

This model is deeper than the earlier ones because it has two hidden layers instead of one.

### What this section adds

This section shows the contrast between:

- manual training with `GradientTape`
- standard Keras workflow with `model.compile()`, `model.fit()`, and `model.evaluate()`

So it serves as a comparison example rather than one of the three main notebook parts.

## Why the Loss and Optimizer Were Chosen

### Optimizer: `Adam`

`Adam` is used because it is a common and effective default optimizer for neural network training. It adapts learning rates automatically and usually converges faster and more stably than plain stochastic gradient descent.

### Loss: `SparseCategoricalCrossentropy`

This loss is appropriate because MNIST is a 10-class classification task and the labels are integer class IDs from `0` to `9`.

In the custom-loop parts, the loss is used as:

- `SparseCategoricalCrossentropy(from_logits=True)`

This is correct because the last layer outputs raw logits, not probabilities.

In the final Functional API section, the output layer uses `softmax`, so the compiled loss is:

- `sparse_categorical_crossentropy`

That version matches probability outputs instead of raw logits.

## Technical Characteristics

- manual training with `tf.GradientTape`
- manual gradient computation and weight updates
- batching with `tf.data.Dataset`
- custom metric tracking
- custom callback definition
- comparison between low-level and high-level Keras workflows
- both `Sequential` and Functional API models
- MNIST multi-class classification

## Packages Used

- `tensorflow`
- `numpy`
- `os`
- `warnings`

Main Keras/TensorFlow components used:

- `tf.keras.datasets.mnist`
- `tf.keras.models.Sequential`
- `tf.keras.models.Model`
- `tf.keras.layers.Input`
- `tf.keras.layers.Flatten`
- `tf.keras.layers.Dense`
- `tf.keras.losses.SparseCategoricalCrossentropy`
- `tf.keras.optimizers.Adam`
- `tf.keras.metrics.SparseCategoricalAccuracy`
- `tf.keras.callbacks.Callback`
- `tf.GradientTape`
- `tf.data.Dataset`

## Summary

This project is a step-by-step demonstration of custom neural network training in TensorFlow/Keras using the MNIST dataset. Part 1 introduces a manual training loop, Part 2 adds accuracy tracking, and Part 3 adds a custom callback for epoch-level logging. The notebook finishes with a standard Functional API classifier to compare custom training with the usual Keras workflow. The project is useful for understanding what happens inside model training rather than only using high-level training commands.
```
