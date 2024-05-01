import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Define a function to normalize the data
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0 - 0.5, label

# Load the MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Prepare and normalize the dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(64)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Extract a batch of unique images
for images, labels in ds_train.take(1):
    unique_images = images.numpy()  # Convert tensor to numpy array

# Display a grid of unique images
fig, axes = plt.subplots(4, 16, figsize=(16, 4), sharex=True, sharey=True)  # Create a 4x16 grid of subplots with a wider figure

for i in range(4):  # Loop over rows
    for j in range(16):  # Loop over columns
        index = i * 16 + j  # Calculate the index in the batch
        axes[i, j].imshow(unique_images[index].squeeze(), cmap='gray')  # Show the image using a grayscale colormap
        axes[i, j].axis('off')  # Turn off axis labels and ticks

plt.savefig("normalize")  # Display the plot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.callbacks import LearningRateScheduler

# Import the 'numpy' library for numerical operations
import numpy as np

# Import the 'functools' module for higher-order functions
import functools

# Import Adam optimizer from TensorFlow
from tensorflow.keras.optimizers import Adam

# Import data loading and transformation utilities
import tensorflow_datasets as tfds

# Import 'tqdm' for creating progress bars during training
import tqdm

# Import 'trange' and 'tqdm' specifically for notebook compatibility
from tqdm import tqdm


# Import the 'matplotlib.pyplot' library for plotting graphs
import matplotlib.pyplot as plt

# TensorFlow does not need an external library like `einops` for tensor rearrangement
# but you can still use it if preferred:
from einops import rearrange
# import keras
# from keras.activations import silu
# Import the 'math' module for mathematical operations
import math
def forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt):
    """
    Parameters:
    - x0: Initial sample value (scalar)
    - noise_strength_fn: Function of time, outputs scalar noise strength
    - t0: Initial time
    - nsteps: Number of diffusion steps
    - dt: Time step size

    Returns:
    - x: Trajectory of sample values over time
    - t: Corresponding time points for the trajectory
    """

    # Initialize the trajectory tensor
    x = tf.Variable(np.zeros(nsteps + 1, dtype=np.float32))

    # Set the initial sample value
    x[0].assign(x0)

    # Generate time points for the trajectory
    t = t0 + tf.range(nsteps + 1, dtype=tf.float32) * dt

    # Perform Euler-Maruyama time steps for diffusion simulation
    for i in range(nsteps):
        # Get the noise strength at the current time
        noise_strength = noise_strength_fn(t[i].numpy())

        # Generate a random normal variable
        random_normal = tf.random.normal(shape=())

        # Update the trajectory using Euler-Maruyama method
        x[i + 1].assign(x[i] + random_normal * noise_strength)

    # Return the trajectory and corresponding time points
    return x.numpy(), t.numpy()
# Example noise strength function: always equal to 1
def noise_strength_constant(t):
    """
    Example noise strength function that returns a constant value (1).

    Parameters:
    - t: Time parameter (unused in this example)

    Returns:
    - Constant noise strength (1)
    """
    return 1
# Number of diffusion steps
nsteps = 100

# Initial time
t0 = 0

# Time step size
dt = 0.1

# Noise strength function
noise_strength_fn = noise_strength_constant

# Initial sample value
x0 = 0

# Number of tries for visualization
num_tries = 5

# Setting larger width and smaller height for the plot
plt.figure(figsize=(15, 3))

# Loop for multiple trials
for i in range(num_tries):

    # Simulate forward diffusion
    x, t = forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt)

    # Plot the trajectory
    plt.plot(t, x, label=f'Trial {i+1}')  # Adding a label for each trial

# Labeling the plot
plt.xlabel('Time', fontsize=20)
plt.ylabel('Sample Value ($x$)', fontsize=20)

# Title of the plot
plt.title('Forward Diffusion Visualization', fontsize=20)

# Adding a legend to identify each trial
plt.legend()

# Show the plot
plt.savefig("forward_diffusion_visualized")
# Reverse diffusion for N steps in 1D.
def reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt):
    """
    Parameters:
    - x0: Initial sample value (scalar)
    - noise_strength_fn: Function of time, outputs scalar noise strength
    - score_fn: Score function
    - T: Final time
    - nsteps: Number of diffusion steps
    - dt: Time step size

    Returns:
    - x: Trajectory of sample values over time
    - t: Corresponding time points for the trajectory
    """

    # Initialize the trajectory array
    x = np.zeros(nsteps + 1)

    # Set the initial sample value
    x[0] = x0

    # Generate time points for the trajectory
    t = np.arange(nsteps + 1) * dt

    # Perform Euler-Maruyama time steps for reverse diffusion simulation
    for i in range(nsteps):

        # Calculate noise strength at the current time
        noise_strength = noise_strength_fn(T - t[i])

        # Calculate the score using the score function
        score = score_fn(x[i], 0, noise_strength, T - t[i])

        # Generate a random normal variable
        random_normal = np.random.randn()

        # Update the trajectory using the reverse Euler-Maruyama method
        x[i + 1] = x[i] + score * noise_strength**2 * dt + noise_strength * random_normal * np.sqrt(dt)

    # Return the trajectory and corresponding time points
    return x, t
# Example score function: always equal to 1
def score_simple(x, x0, noise_strength, t):
    """
    Parameters:
    - x: Current sample value (scalar)
    - x0: Initial sample value (scalar)
    - noise_strength: Scalar noise strength at the current time
    - t: Current time

    Returns:
    - score: Score calculated based on the provided formula
    """

    # Calculate the score using the provided formula
    score = - (x - x0) / ((noise_strength**2) * t)

    # Return the calculated score
    return score
# Number of reverse diffusion steps
nsteps = 100

# Initial time for reverse diffusion
t0 = 0

# Time step size for reverse diffusion
dt = 0.1

# Function defining constant noise strength for reverse diffusion
noise_strength_fn = noise_strength_constant

# Example score function for reverse diffusion
score_fn = score_simple

# Initial sample value for reverse diffusion
x0 = 0

# Final time for reverse diffusion
T = 11

# Number of tries for visualization
num_tries = 5

# Setting larger width and smaller height for the plot
plt.figure(figsize=(15, 3))

# Loop for multiple trials
for i in range(num_tries):
    # Draw from the noise distribution, which is diffusion for time T with noise strength 1
    x0 = np.random.normal(loc=0, scale=T)

    # Simulate reverse diffusion
    x, t = reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt)

    # Plot the trajectory
    plt.plot(t, x, label=f'Trial {i+1}')  # Adding a label for each trial

# Labeling the plot
plt.xlabel('Time', fontsize=20)
plt.ylabel('Sample Value ($x$)', fontsize=20)

# Title of the plot
plt.title('Reverse Diffusion Visualized', fontsize=20)

# Adding a legend to identify each trial
plt.legend()

# Show the plot
plt.savefig("reverse_diff_visualized")
class GaussianFourierProjection(tf.keras.layers.Layer):
    def __init__(self, embed_dim, scale=30.):
        """
        Parameters:
        - embed_dim: Dimensionality of the embedding (output dimension)
        - scale: Scaling factor for random weights (frequencies)
        """
        super(GaussianFourierProjection, self).__init__()

        # Randomly sample weights (frequencies) during initialization.
        # These weights (frequencies) are fixed during optimization and are not trainable.
        self.W = tf.Variable(tf.random.normal([embed_dim // 2]) * scale, trainable=False)

    def call(self, x):
        """
        Parameters:
        - x: Input tensor representing time steps

        Applies the layer logic and returns the output.
        """
        # Calculate the cosine and sine projections: Cosine(2 pi freq x), Sine(2 pi freq x)
        x_proj = x[:, tf.newaxis] * self.W[tf.newaxis, :] * 2 * np.pi#tf.expand_dims(x, axis=-1) * tf.expand_dims(self.W, axis=0) * 2 * np.pi

        # Concatenate the sine and cosine projections along the last dimension
        return tf.concat([tf.sin(x_proj), tf.cos(x_proj)], axis=-1)
class Dense(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        """
        Parameters:
        - input_dim: Dimensionality of the input features
        - output_dim: Dimensionality of the output features
        """
        super(Dense, self).__init__()

        # Define a fully connected (dense) layer
        self.dense = tf.keras.layers.Dense(output_dim, input_shape=(input_dim,), trainable=True)

    def call(self, x):
        """
        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor after passing through the fully connected layer
          and reshaping to a 4D tensor (feature map)
        """

        # Apply the fully connected layer
        x = self.dense(x)

        # Reshape the output to a 4D tensor, adding singleton dimensions for height and width
        return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        # return tf.reshape(x, [-1, self.dense.units, 1, 1])


class UNet(tf.keras.Model):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        super(UNet, self).__init__()
        self.time_embed = tf.keras.Sequential([
            GaussianFourierProjection(embed_dim=embed_dim),
            tf.keras.layers.Dense(embed_dim, activation=None)#DenseToFeatureMap(embed_dim, embed_dim)
        ])
        self.conv1 = tf.keras.layers.Conv2D(channels[0], 3, strides=1, padding='same', use_bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = tf.keras.layers.GroupNormalization(groups=4,epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(channels[1], 3, strides=4, padding='same', use_bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = tf.keras.layers.GroupNormalization(groups=32,epsilon=1e-5)

        self.conv3 = tf.keras.layers.Conv2D(channels[2], 3, strides=1, padding='same', use_bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = tf.keras.layers.GroupNormalization(groups=32,epsilon=1e-5)

        self.conv4 = tf.keras.layers.Conv2D(channels[3], 3, strides=1, padding='same', use_bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = tf.keras.layers.GroupNormalization(groups=32,epsilon=1e-5)

        # Decoding layers where the resolution increases
        self.tconv4 = tf.keras.layers.Conv2DTranspose(channels[2], 3, strides=1, padding='same', use_bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = tf.keras.layers.GroupNormalization(groups=32,epsilon=1e-5)
        self.tconv3 = tf.keras.layers.Conv2DTranspose(channels[1], 3, strides=1, padding='same', use_bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = tf.keras.layers.GroupNormalization(groups=32,epsilon=1e-5)
        self.tconv2 = tf.keras.layers.Conv2DTranspose(channels[0], 3, strides=4, padding='same', use_bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = tf.keras.layers.GroupNormalization(groups=32,epsilon=1e-5)
        self.tconv1 = tf.keras.layers.Conv2DTranspose(1, 3, strides=1, padding='same',use_bias = False)

        # The swish activation function
        self.act = lambda x: x * tf.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def call(self, x, t, y=None):
        # print(f"input shape:{x.shape}")
        embed = self.act(self.time_embed(t))

        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        # print(f"h1 shape:{h1.shape}")
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        # print(f"h2 shape:{h2.shape}")
        # Additional encoding path layers (copied from the original code)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        # print(f"h3 shape:{h3.shape}")
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))
        # print(f"h4 shape:{h4.shape}")
        # Decoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)

        h = self.act(self.tgnorm4(h))
        # print(f"h shape after 1st deconv:{h.shape}")
        h = self.tconv3(tf.concat([h, h3], axis=-1))
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        # print(f"h shape after 2nd deconv:{h.shape}")
        h = self.tconv2(tf.concat([h, h2], axis=-1))
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        # print(f"h shape after 3rd deconv:{h.shape}")
        h = self.tconv1(tf.concat([h, h1], axis=-1))
        # print(f"h shape after last deconv:{h.shape}")

        # Normalize output
        h = h / tf.expand_dims(tf.expand_dims(tf.expand_dims(self.marginal_prob_std(t), axis=-1), axis=-1),axis = -1)
        return h
class UNetRes(tf.keras.Model):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        super(UNetRes, self).__init__()
        self.time_embed = tf.keras.Sequential([
            GaussianFourierProjection(embed_dim=embed_dim),
            tf.keras.layers.Dense(embed_dim, activation=None)#DenseToFeatureMap(embed_dim, embed_dim)
        ])
        self.conv1 = tf.keras.layers.Conv2D(channels[0], 3, strides=1, padding='same', use_bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)

        self.conv2 = tf.keras.layers.Conv2D(channels[1], 3, strides=4, padding='same', use_bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)

        self.conv3 = tf.keras.layers.Conv2D(channels[2], 3, strides=1, padding='same', use_bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)

        self.conv4 = tf.keras.layers.Conv2D(channels[3], 3, strides=1, padding='same', use_bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)

        # Decoding layers where the resolution increases
        self.tconv4 = tf.keras.layers.Conv2DTranspose(channels[2], 3, strides=1, padding='same', use_bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)
        self.tconv3 = tf.keras.layers.Conv2DTranspose(channels[1], 3, strides=1, padding='same', use_bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)
        self.tconv2 = tf.keras.layers.Conv2DTranspose(channels[0], 3, strides=4, padding='same', use_bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = tf.keras.layers.LayerNormalization(axis=-1, center=True, scale=True)
        self.tconv1 = tf.keras.layers.Conv2DTranspose(1, 3, strides=1, padding='same',use_bias = False)

        # The swish activation function
        self.act = lambda x: x * tf.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def call(self, x, t, y=None):
        embed = self.act(self.time_embed(t))

        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # Decoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))
        h = self.tconv3(h + h3)
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self.tconv2(h + h2)
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self.tconv1(h + h1)

        # Normalize output
        h = h / tf.expand_dims(tf.expand_dims(tf.expand_dims(self.marginal_prob_std(t), axis=-1), axis=-1),axis = -1)
        return h

# Set the GPU as the device if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Marginal Probability Standard Deviation Function
def marginal_prob_std(t, sigma):
    """
    Compute the standard deviation of $p_{0t}(x(t) | x(0))$.

    Parameters:
    - t: A vector of time steps.
    - sigma: The $\sigma$ in our SDE.

    Returns:
    - The standard deviation as a TensorFlow tensor.
    """
    # Convert time steps to a TensorFlow tensor
    t = tf.convert_to_tensor(t, dtype=tf.float32)

    # Calculate and return the standard deviation based on the given formula
    return tf.sqrt((sigma**(2 * t) - 1.) / (2. * tf.math.log(sigma)))
# Ensure that GPUs are available and set for TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to only allocate memory as needed on the GPU, not all at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set at program startup
        print(f'Error: {e}')

# Define the functions as previously described
def marginal_prob_std(t, sigma):
    """
    Compute the standard deviation of $p_{0t}(x(t) | x(0))$ based on the SDE.

    Parameters:
    - t: A vector of time steps.
    - sigma: The $\sigma$ in our SDE.

    Returns:
    - The standard deviation as a TensorFlow tensor.
    """
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    return tf.sqrt((sigma**(2 * t) - 1.) / (2. * tf.math.log(sigma)))

def diffusion_coeff(t, sigma):
    """
    Compute the diffusion coefficient of our SDE.

    Parameters:
    - t: A vector of time steps.
    - sigma: The $\sigma$ in our SDE.

    Returns:
    - The vector of diffusion coefficients as a TensorFlow tensor.
    """
    t = tf.convert_to_tensor(t, dtype=tf.float32)
    return tf.pow(sigma, t)

# Sigma Value
sigma =  25.0

# Partial functions setup with sigma value
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """
    The loss function for training score-based generative models in TensorFlow,
    revised to match the PyTorch version more closely.

    Parameters:
    - model: A TensorFlow model instance that represents a time-dependent score-based model.
    - x: A mini-batch of training data.
    - marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    - eps: A tolerance value for numerical stability.
    """
    # Sample time uniformly in the range (eps, 1-eps)
    random_t = tf.random.uniform((tf.shape(x)[0],), minval=eps, maxval=1-eps)

    # Find the noise std at the sampled time `t`
    std = marginal_prob_std(random_t)

    # Generate normally distributed noise
    z = tf.random.normal(tf.shape(x))

    # Perturb the input data with the generated noise
    perturbed_x = x + z * tf.reshape(std, (-1, 1, 1, 1))  # Explicit reshaping for broadcasting

    # Get the score from the model using the perturbed data and time
    score = model(perturbed_x, random_t)

    # Calculate the loss based on the score and noise
    loss = tf.reduce_mean(tf.reduce_sum((score * tf.reshape(std, (-1, 1, 1, 1)) + z)**2, axis=[1, 2, 3]))

    return loss
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           x_shape=(28,28,1),  # Adjusted shape for TensorFlow's default channel-last data format
                           num_steps=500,
                           device='GPU',  # Use 'GPU' or 'CPU'
                           eps=1e-3, y=None):
    """
    Generate samples from score-based models with the Euler-Maruyama solver in TensorFlow.

    Parameters:
    - score_model: A TensorFlow model that represents the time-dependent score-based model.
    - marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
    - diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    - batch_size: The number of samplers to generate by calling this function once.
    - x_shape: The shape of the samples.
    - num_steps: The number of sampling steps, equivalent to the number of discretized time steps.
    - device: 'GPU' for running on GPUs, and 'CPU' for running on CPUs.
    - eps: The smallest time step for numerical stability.
    - y: Target tensor (not used in this function).

    Returns:
    - Samples.
    """

    # Device setting in TensorFlow
    if device == 'GPU':
        tf_device = '/device:GPU:0'
    else:
        tf_device = '/device:CPU:0'

    with tf.device(tf_device):
        # Initialize time and the initial sample
        t = tf.ones((batch_size,))  # Uniform time initialized at 1
        init_x = tf.random.normal((batch_size, *x_shape)) * marginal_prob_std(t)[:, tf.newaxis, tf.newaxis, tf.newaxis]

        # Generate time steps
        time_steps = tf.linspace(1.0, eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        x = init_x

        # Sample using Euler-Maruyama method
        for time_step in tqdm(time_steps):
            batch_time_step = tf.fill([batch_size], time_step)
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, tf.newaxis, tf.newaxis, tf.newaxis] * score_model(x, batch_time_step, y=y) * step_size
            x = mean_x + tf.sqrt(step_size) * g[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.random.normal(tf.shape(x))

        # Do not include any noise in the last sampling step.
        return mean_x

# # Define the score-based model
score_model = UNet(marginal_prob_std=marginal_prob_std_fn)  # Adjusted to your model's class name
score_model.compile(optimizer=Adam(learning_rate=1e-3))  # Adjusted learning rate

n_epochs = 15
batch_size = 512
lr = 5e-4 #9.8e-04

# Load the MNIST dataset for training
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.astype('float32') / 255
# x_train = np.expand_dims(x_train, -1)  # Adjust for TensorFlow's default channel-last data format
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))  # Dummy y_train as x_train just for batching purposes
train_dataset = train_dataset.shuffle(60000).batch(batch_size)

# Initialize the Adam optimizer with the specified learning rate

# Learning rate scheduler to adjust the learning rate during training
def lr_schedule(epoch,lr,total_epochs):

    """
    Function to update learning rate based on epoch.

    Args:
        epoch (int): Current epoch number.
        initial_lr (float): Initial learning rate.
        final_lr (float): Final learning rate.
        total_epochs (int): Total number of epochs.

    Returns:
        float: Updated learning rate for the given epoch.
    """
    lr = lr - (epoch * (lr) / total_epochs)
    return lr

# Training loop over epochs
for epoch in range(n_epochs):
    total_loss = 0
    num_iter = 0
    for x, y in train_dataset:
        with tf.GradientTape() as tape:
            # Forward pass
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            # print(loss)
        # Compute gradients
        gradients = tape.gradient(loss, score_model.trainable_variables)
        # Update weights
        score_model.optimizer.apply_gradients(zip(gradients, score_model.trainable_variables))
        # Update average loss
        total_loss +=loss * x.shape[0]
        num_iter += x.shape[0]

    # Adjust the learning rate
    # lr_current = lr_schedule(epoch,lr,n_epochs)
    # score_model.optimizer.learning_rate = lr_current

    # Print average loss and learning rate for the current epoch
    print('Epoch {} / {}: Average Loss: {:.5f} '.format(epoch+1, n_epochs, total_loss / num_iter))

    # Save the model checkpoint after each epoch of training
    score_model.save_weights('ckpt_res.h5')


# Load the pre-trained checkpoint from disk.
# Assuming the model weights are saved in a TensorFlow checkpoint format
ckpt_path = 'ckpt_res.h5'
ckpt = tf.train.Checkpoint(model=score_model)
ckpt.restore(tf.train.latest_checkpoint(ckpt_path))

# Set sample batch size and number of steps
sample_batch_size = 64
num_steps = 500

# Choose the Euler-Maruyama sampler
sampler = Euler_Maruyama_sampler

# Generate samples using the specified sampler
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device='/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0',
                  y=None)

# Clip samples to be in the range [0, 1]
samples = tf.clip_by_value(samples, 0.0, 1.0)

# Visualize the generated samples
import matplotlib.pyplot as plt
for i in range(sample_batch_size):
    plt.subplot(8, 8, i + 1)
    subplot= tf.keras.preprocessing.image.array_to_img(samples[i], data_format='channels_last')
    plt.axis('off')
    plt.imsave(f'image_{i}.png', subplot)


# Load the pre-trained checkpoint from disk.
# Assuming the model weights are saved in a TensorFlow checkpoint format
ckpt_path = 'ckpt_res.h5'
ckpt = tf.train.Checkpoint(model=score_model)
ckpt.restore(tf.train.latest_checkpoint(ckpt_path))

# Set sample batch size and number of steps
sample_batch_size = 64
num_steps = 500

# Choose the Euler-Maruyama sampler
sampler = Euler_Maruyama_sampler

# Generate samples using the specified sampler
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  num_steps=num_steps,
                  device='/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0',
                  y=None)

# Clip samples to be in the range [0, 1]
samples = tf.clip_by_value(samples, 0.0, 1.0)

# Visualize the generated samples
import matplotlib.pyplot as plt
for i in range(sample_batch_size):
    plt.subplot(8, 8, i + 1)
    subplot= tf.keras.preprocessing.image.array_to_img(samples[i], data_format='channels_last')
    plt.imshow(subplot, cmap='gray')  # Assuming grayscale images, change the cmap if needed
    plt.axis('off')
plt.savefig(f'result.png', cmap='gray')