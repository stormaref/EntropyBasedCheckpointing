
# Entropy-Based Checkpointing

This repository contains an implementation of entropy-based checkpointing for model training. Unlike traditional methods that rely on validation accuracy to save the best model, this approach saves the model when it achieves the **highest entropy** during training or validation. This makes it particularly useful in noisy or complex environments, where the model's uncertainty can be a more reliable indicator of generalization performance.

## Purpose

Entropy-based checkpointing is designed to handle scenarios where noisy labels or complex datasets might cause a model to overfit. By focusing on entropy (which measures the uncertainty of predictions), this method prioritizes saving the model when it demonstrates balanced uncertainty, potentially indicating better generalization.

### Key Features
- **Entropy Calculation**: Tracks the entropy of the two highest probabilities for each prediction.
- **Checkpointing**: Saves the model when it achieves the maximum entropy recorded so far during either training or validation.
- **Easy Integration**: Can be easily added to any PyTorch model training loop.

## Installation

To use the entropy-based checkpointing class, ensure you have PyTorch installed:

```bash
pip install torch
```

Then, include the `EntropyCheckPoint` class in your project.

## Usage

1. **Initialize the Checkpoint Class**: Provide your model and optimizer to the `EntropyCheckPoint` class.

```python
checkpoint = EntropyCheckPoint(model, optimizer)
```

2. **During Training or Validation**: After each batch, append the model’s output to the checkpoint for entropy calculation.

```python
# Inside training/validation loop
checkpoint.append_output(output)
```

3. **After Each Epoch**: Calculate the entropy and save the model if it surpasses the previous best.

```python
# After training/validation phase
checkpoint.save_checkpoint(epoch)
checkpoint.reset_output()  # Reset for the next epoch
```

4. **Loading the Best Model**: After training, load the model from the checkpoint file.

```python
checkpoint.load_checkpoint()
```

### Example

Below is a simple example of how you would integrate this into a training loop:

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        checkpoint.append_output(outputs)  # Save output for entropy calculation

    # Validation phase
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            checkpoint.append_output(outputs)

    # Save model if the entropy is the highest
    checkpoint.save_checkpoint(epoch)
    checkpoint.reset_output()  # Ready for next epoch
```

## Parameters

- **model**: The PyTorch model being trained.
- **optimizer**: The optimizer used for training.
- **filename**: (Optional) The file where the best model will be saved (default: `'entropy_checkpoint.pth'`).

## Additional Notes

- The entropy-based method can be applied during both training and validation phases, and can be an alternative to traditional accuracy-based checkpointing, especially for noisy datasets.
- This method works well in environments where the model's uncertainty is a critical factor in its generalization performance.
  
