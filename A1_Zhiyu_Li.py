import numpy as np

# Set the random seed for reproducibility
np.random.seed(0)

# load the data
fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# Layer


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize weights & biases.
        Weights should be initialized with values drawn from a normal
        distribution scaled by 0.01.
        Biases are initialized to 0.0.
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        A forward pass through the layer to give z.
        Compute it using np.dot(...) and then add the biases.
        """
        self.inputs = inputs
        self.z = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dz):
        """
        Backward pass
        """
        # Gradients of weights
        self.dweights = np.dot(self.inputs.T, dz)
        # Gradients of biases
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        # Gradients of inputs
        self.dinputs = np.dot(dz, self.weights.T)

# Activations
# Relu


class ReLu:
    """
    ReLu activation
    """

    def forward(self, z):
        """
        Forward pass
        """
        self.z = z
        self.activity = np.maximum(0, z)

    def backward(self, dactivity):
        """
        Backward pass
        """
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0


# Softmax
class Softmax:
    def forward(self, z):
        """
        """
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)

        return self.probs

    def backward(self, dprobs):
        """
        """
        # Empty array
        self.dz = np.empty_like(dprobs)

        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            # flatten to a column vector
            prob = prob.reshape(-1, 1)
            # Jacobian matrix
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)

# Loss function
# Crossentropy loss


class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # clip to prevent division by 0
        # clip both sides to not bias up.
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        # negative log likelihoods
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)

        return loss.mean(axis=0)

    def backward(self, probs, oh_y_true):
        """
        Use one-hot encoded y_true.
        """
        # Number of examples in batch and number of classes
        batch_sz, n_class = probs.shape
        # get the gradient
        self.dprobs = -oh_y_true / probs
        # normalize the gradient
        self.dprobs = self.dprobs / batch_sz

# Optimizer
# Stochastic Gradient Descent


class SGD:
    """
    """

    def __init__(self, learning_rate=1.0):
        # Initialize the optimizer with a learning rate
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

# Helper functions
# Covert probabilities to predictions


def predictions(probs):
    """
    """
    y_preds = np.argmax(probs, axis=1)
    return y_preds

# Accuracy


def accuracy(y_preds, y_true):
    """
    """
    return np.mean(y_preds == y_true)


# One-hot encoding
n_class = np.unique(y_train).shape[0]
oh_y_true = np.eye(n_class)[y_train]


def forward_pass(X, y_true, oh_y_true):
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    dense3.forward(activation2.activity)
    probs = output_activation.forward(dense3.z)
    loss = loss_function.forward(probs, oh_y_true)
    return probs, loss


def backward_pass(probs, y_true, oh_y_true):
    loss_function.backward(probs, oh_y_true)
    output_activation.backward(loss_function.dprobs)
    dense3.backward(output_activation.dz)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)


# Initialize the network and set hyperparameters
epochs = 10
n_batch = 20
n_inputs = X_train.shape[1]
n_neurons1 = 4
n_neurons2 = 8
batch_size = len(X_train) // n_batch

dense1 = DenseLayer(n_inputs, n_neurons1)
activation1 = ReLu()
dense2 = DenseLayer(n_neurons1, n_neurons2)
activation2 = ReLu()
dense3 = DenseLayer(n_neurons2, n_class)
output_activation = Softmax()
loss_function = CrossEntropyLoss()
optimizer = SGD()

# Training loop
for epoch in range(epochs):
    print('Epoch:', epoch)
    for batch_i in range(n_batch):
        # Get a mini-batch of data from X_train and y_train
        if batch_i == 0:
            start_idx = batch_i * batch_size
            end_idx = (batch_i + 1) * batch_size
        else:
            start_idx = batch_i * batch_size + 1
            end_idx = (batch_i + 1) * batch_size

        # One-hot encode y_true
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        oh_y_batch = np.eye(n_class)[y_batch]

        # Forward pass
        probs, loss = forward_pass(X_batch, y_batch, oh_y_batch)

        # Print accuracy and loss
        y_preds = predictions(probs)
        acc = accuracy(y_preds, y_batch)
        print('Batch:', "{:.0f}".format(batch_i), '\tAccuracy:',
              "{:.2f}".format(acc), '\t\tLoss:', "{:.2f}".format(loss))

        # Backward pass
        backward_pass(probs, y_batch, oh_y_batch)

        # Update the weights
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)

# Test with the test data
probs_test, loss_test = forward_pass(X_test, y_test,
                                     np.eye(n_class)[y_test])

# Calculate accuracy on the test set
y_preds_test = predictions(probs_test)
acc_test = accuracy(y_preds_test, y_test)
print('\n ------- Test Accuracy:', "{:.3f}".
      format(acc_test), "-------")
print('\n ------- Test Loss:', "{:.3f}".
      format(loss_test), "-------")
