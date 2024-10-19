import torch
import torch.nn.functional as F

class EntropyCheckPoint:
    def __init__(self, model, optimizer, filename='entropy_checkpoint.pth'):
        self.model = model
        self.optimizer = optimizer
        self.filename = filename
        self.max_entropy = None
        self.best_epoch = None
        self.outputs = []  # Store outputs for entropy calculation

    def reset_output(self):
        self.outputs = []  # Reset outputs for the next validation phase

    def append_output(self, output):
        self.outputs.append(output)  # Append the output to the list

    def calculate_entropy(self):
        all_outputs = torch.cat(self.outputs)  # Concatenate all outputs
        probabilities = F.softmax(all_outputs, dim=1)  # Apply softmax

        # Get the two highest probabilities and calculate entropy
        top_probs, _ = torch.topk(probabilities, 2, dim=1)
        entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10), dim=1)  # Avoid log(0)
        
        return entropy.mean()  # Return mean entropy for all batches

    def save_checkpoint(self, epoch):
        # Calculate entropy using the appended outputs
        calculated_entropy = self.calculate_entropy()

        # Save the model if the current entropy is greater than the max recorded entropy
        if self.max_entropy is None or calculated_entropy > self.max_entropy:
            self.max_entropy = calculated_entropy
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.filename)
            print(f'Saved new model at epoch {epoch+1} with entropy: {calculated_entropy.item()}')

    def load_checkpoint(self):
        print(f'Loading model from best epoch: {self.best_epoch + 1}')
        self.model.load_state_dict(torch.load(self.filename))