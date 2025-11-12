
# Everything about NNA.

 ![class](/ima/ima1.png)

---------------------------------------------

**Repository summary**

1.  **Intro** 🧳

2.  **Tech Stack** 🤖

3.  **Features** 🤳🏽

4.  **Process** 👣

5.  **Learning** 💡

6.  **Improvement** 🔩


7.  **Running the Project** ⚙️

8 .  **More** 🙌🏽



---------------------------------------------

# :computer: Everything about NNA.  :computer:

---------------------------------------------

# I. Let's talk about NNA.

Understanding artificial neural networks begins with understanding why they exist. The purpose of this repository is to create a complete and accessible space for studying ANNs, combining theory, history, and practice. It is meant to be both academic and experimental, a place where concepts are explained with precision but also brought to life through code.

The field of neural networks has evolved for decades, driven by one simple idea: machines can learn patterns from experience. Behind every neural model there is a system that tries to capture the logic of the human brain, simplified into mathematics, data, and algorithms. This repository collects that journey — from the origins of the first perceptrons to the complex deep architectures that now power vision, language, and reasoning systems.

 ![class](/ima/ima2.png)

 General Introduction

Artificial neural networks, or ANNs, are mathematical models inspired by how neurons in the human brain communicate and process information. Each neuron receives inputs, processes them, and sends an output to other neurons. When these units are connected in layers, they can collectively learn patterns that are too complex for simple algorithms.

This repository was born from the need to document that process — not only to explain how ANNs work but also to make them tangible through examples, equations, and Python implementations. It aims to become a reliable reference that connects the abstract logic of mathematics with the hands-on experience of coding and experimentation.

The intention is not to build a tutorial, but rather a knowledge map that organizes the main ideas, principles, and mechanisms that make neural networks function. Every section was written to encourage understanding, not memorization, so the concepts flow naturally from the basics to more advanced topics.

⸻

Purpose of this Repository

The main goal of this project is to gather the essential knowledge around ANNs into a single, structured space. It includes theoretical explanations, mathematical intuition, historical context, and practical examples implemented in Python using frameworks like PyTorch and TensorFlow.

The repository is designed to be modular and self-contained. Each folder represents a conceptual step — starting with fundamental definitions, continuing with theoretical details, and finally leading to implementation and real-world applications.

Beyond providing information, this repository serves as a reflection tool. It is meant to help learners and professionals think critically about how neural networks operate, how they fail, and how they can be improved.

It also recognizes that learning neural networks is not a static process. New architectures, optimizers, and learning methods appear constantly. That is why this space is meant to evolve, just like the field itself. Every update, new example, or added note is part of the continuous process of learning and rediscovery.

⸻

How This Material Is Organized

The structure of the repository was carefully designed to follow the logical growth of understanding. It begins with an introduction that explains where ANNs come from, followed by theoretical sections that describe their inner mechanics, and finally progresses toward architectures, applications, and resources for deeper study.

Readers can approach the material in two different ways.
They can follow the linear order — from history to applications — as a complete learning path, or they can explore specific folders depending on their goals. For example, one might go directly to convolutional networks for vision, or to recurrent networks for text and sequence prediction.

Each part of the repository mirrors the evolution of the field itself. The introduction explains the origins, the theory shows the mathematical engine, the taxonomy classifies the architectures, and the applications demonstrate what those systems can achieve. Together, they form a coherent narrative of how neural networks moved from biological inspiration to digital intelligence.

# II. What are Artificial Neural Networks (ANNs).

Understanding what an artificial neural network truly is requires looking at both its inspiration and its evolution. ANNs did not appear suddenly. They were born from a long tradition of research that tried to answer one fundamental question: can a machine learn like a brain?
To explore that question, we begin with their biological roots, follow their historical milestones, define what they are in mathematical terms, and finally explain how they learn.

⸻

Biological Inspiration

Artificial neural networks were inspired by the way biological neurons transmit and transform signals inside the brain. A biological neuron receives electrical impulses through its dendrites, processes them in the soma, and emits a signal through the axon. When thousands of neurons interact, they create complex patterns that give rise to perception, memory, and reasoning.

Artificial neurons mimic that same mechanism, but with mathematical components. Instead of electrical impulses, they receive numerical inputs. Instead of synapses, they use weights. Each neuron multiplies its inputs by these weights, sums the results, and applies a nonlinear activation function to produce an output.

When connected together, these artificial neurons form layers. The first layer receives raw data (like pixels or words), the hidden layers extract abstract features, and the final layer delivers predictions or classifications. This simple principle — composition through layers — is what allows ANNs to approximate almost any function, from recognizing faces to predicting language.

⸻

A Brief History of Development

The story of neural networks began in the 1940s, when Warren McCulloch and Walter Pitts proposed a simplified model of a neuron that could perform logical operations. A decade later, Frank Rosenblatt introduced the Perceptron, the first true learning algorithm capable of adjusting its own weights based on experience.

However, enthusiasm faded during the 1970s after Marvin Minsky and Seymour Papert showed that the Perceptron could not solve nonlinear problems, such as distinguishing between overlapping patterns. This period, known as the AI winter, slowed progress for years.

The revival came in the 1980s with the rediscovery of the backpropagation algorithm, which made it possible to train networks with multiple hidden layers — the Multilayer Perceptron (MLP). This innovation reopened the field and set the stage for modern deep learning.

In the following decades, advances in computing power, large datasets, and specialized hardware (like GPUs) allowed neural networks to expand dramatically. By the 2010s, architectures such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) were dominating computer vision and natural language processing.

Today, we are in the era of Transformers, a paradigm that uses attention mechanisms to process information in parallel and at scale. The journey from the simple perceptron to modern models like GPT or BERT reflects how mathematical ideas can evolve into systems capable of generating language, art, and insight.

⸻

Conceptual Definition

An artificial neural network is a computational model designed to recognize patterns through learning. It is composed of layers of interconnected nodes, or neurons, where each connection carries a weight that determines the strength and direction of influence between two units.

In formal terms, an ANN defines a function f(x; θ) that maps an input x to an output y, where θ represents all the parameters (weights and biases) that the network learns during training. Through optimization, the model adjusts these parameters to minimize the difference between its predictions and the correct answers.

What makes neural networks powerful is their ability to learn representations. Instead of relying on manually defined features, they automatically discover the most relevant characteristics of the data. Each layer transforms the input into a more abstract version, gradually building a hierarchy of meaning.

A neural network can be simple — like a few connected neurons for linear regression — or deep, with dozens or hundreds of layers capable of modeling highly nonlinear relationships. Regardless of size, the same logic applies: weighted sums, activations, and iterative learning through error correction.

⸻

How ANNs Learn

Learning in a neural network is the process by which it improves its predictions through experience. It starts with random weights, processes data through forward propagation, computes an error by comparing the output with the expected result, and then adjusts the weights using backward propagation.

The central mechanism is the loss function, which measures how far the prediction is from the desired value. The smaller the loss, the better the model performs. Using optimization algorithms such as Stochastic Gradient Descent (SGD) or Adam, the network updates its parameters in the direction that reduces the loss the most.

There are three main learning paradigms.
In supervised learning, the model learns from labeled data — for instance, associating images with their categories.
In unsupervised learning, it identifies patterns or clusters in data without explicit labels, as in autoencoders or clustering networks.
And in reinforcement learning, the model interacts with an environment and improves through rewards and penalties, similar to how humans learn by experience.

Through repetition, feedback, and optimization, ANNs gradually transform raw information into structured knowledge. This is what allows them not just to store data, but to generalize — to recognize patterns they have never seen before.

# III. Components of an ANN Analysis

Before building or training any neural network, it is essential to understand the components that make it work. Every network, regardless of its scale or complexity, is built upon the same foundation: neurons, layers, weights, activations, and learning mechanisms. These elements interact to transform raw data into structured predictions through continuous adaptation.

This section walks through those internal components, explaining their roles and how they shape the learning process.

⸻

The Internal Structure of a Network

At its most basic level, an artificial neural network is a sequence of layers that process data step by step. Each layer contains neurons, and each neuron performs a simple yet powerful operation: it combines its inputs, adjusts them with weights, adds a bias, and then applies an activation function to produce an output.

Formally, a neuron can be expressed as:

$$
y = f\left(\sum_{i} w_i x_i + b\right)
$$

where x_i are the inputs, w_i are the weights, b is the bias term, and f is the activation function.

The first layer (input layer) receives the external data, while the hidden layers extract patterns and relationships that are not immediately visible. The output layer then translates these learned representations into predictions or classifications.

The power of neural networks comes from combining many of these layers. Each one captures a different level of abstraction — from edges in an image to shapes, objects, and even meaning. The deeper the network, the more complex the representations it can build.

Every connection in the network carries information forward. During training, the network adjusts the weights on those connections to improve its performance. Learning is essentially this gradual redistribution of importance among the connections.

⸻

Activation Functions

Activation functions are the nonlinear heart of neural networks. They decide whether a neuron should activate, and how strongly its signal should influence the next layer. Without them, no matter how many layers we stacked, the entire system would behave like a single linear model.

The sigmoid function was one of the first to be used. It squashes values between 0 and 1, allowing the model to express probabilities:

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

Another common choice is the hyperbolic tangent (tanh), which centers the output between –1 and 1:

$$
f(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

Later, the Rectified Linear Unit (ReLU) became the default in deep learning because of its simplicity and efficiency:

$$
f(x) = \max(0, x)
$$

Variants such as Leaky ReLU, ELU, and GELU were developed to improve stability and prevent inactive neurons. The choice of activation function can influence both how fast a network learns and how stable that learning process remains over time.

⸻

The Training Process

Learning in an ANN happens through a process called backpropagation, where the model repeatedly adjusts its weights to minimize error. It begins with forward propagation, in which data moves through the network to produce an output. Then, the loss function compares that output with the correct value, producing an error signal.

The error is propagated backward, and each weight is updated according to its contribution to the error. This is done using the gradient of the loss with respect to each parameter.

In simple terms, each training step follows the rule:

$$
w_{ij}^{(t+1)} = w_{ij}^{(t)} - \eta \frac{\partial L}{\partial w_{ij}}
$$

where w_{ij} is a weight, \eta is the learning rate, and L is the loss function.

Through this process, the model gradually moves toward a configuration of weights that minimizes the overall loss. After many iterations, the network “learns” to produce accurate outputs for new, unseen data.

⸻

Optimizers and Regularization

Optimizers control how the model updates its parameters. The simplest one, Stochastic Gradient Descent (SGD), updates weights in the direction opposite to the gradient of the loss. More advanced methods like Adam, RMSProp, or Adagrad adjust learning rates dynamically for each parameter, speeding up convergence and improving stability.

Regularization methods ensure that the model does not memorize the training data. Techniques such as Dropout, Early Stopping, and Weight Decay help the network generalize by discouraging overfitting. Regularization can be interpreted as a way to teach the model humility — to learn the underlying structure of the data rather than every single detail.

⸻

Model Evaluation

Once a neural network has been trained, its performance must be measured objectively. Evaluation involves comparing its predictions with ground-truth data using metrics that depend on the task.

For classification problems, accuracy, precision, recall, and the F1-score are common indicators. For regression tasks, mean squared error (MSE) and mean absolute error (MAE) are typically used.

Beyond these numbers, a well-evaluated model should also be interpretable and stable. Understanding why a network made a decision can be as important as the decision itself. In practice, this often requires analyzing activations, visualizing learned filters, and studying how errors propagate across layers.

⸻

Together, these components form the anatomy of every artificial neural network. They define how data enters, how it transforms, and how learning occurs. Mastering these inner mechanisms is the foundation for understanding the architectures and applications that come next.

# IV. Before Conducting an ANN Analysis

Building a neural network is not just about writing code.
Before any model can learn, its data must be understood, structured, and prepared with care.
The quality of this preparation determines the success of the entire analysis.

A neural network is like a lens. If the data is blurry or distorted, even the most advanced architecture will fail to see clearly.
This section explains what should happen before training begins: preparing the data, choosing the right model, defining hyperparameters, and validating the results.

⸻

Data Preparation

The first step in any neural network project is to ensure that the data is clean, consistent, and suitable for training.
Neural networks are extremely sensitive to data quality. A few outliers, missing values, or inconsistent scales can disrupt learning completely.

To mitigate that, several preprocessing steps are applied. Inputs are usually normalized or standardized to keep values within similar ranges, which stabilizes gradient updates. For instance, one may standardize a feature x using:

$$
x’ = \frac{x - \mu}{\sigma}
$$

where \mu is the mean and \sigma the standard deviation of the feature.

Categorical data must be converted into numerical form, often through one-hot encoding or embedding vectors. Outliers should be treated or removed, and missing values must be imputed carefully.

The dataset is then divided into training, validation, and test sets.
The training set teaches the model, the validation set guides hyperparameter tuning, and the test set provides an unbiased estimate of performance.
Without this separation, there is no true way to measure whether a network has learned or merely memorized.

⸻

Model Selection

Choosing the right type of network is as important as cleaning the data.
Not all neural architectures are suited for every problem. The structure must reflect the nature of the input and the desired outcome.

For example, Feedforward Networks work well for tabular or structured data.
Convolutional Networks (CNNs) excel at recognizing spatial relationships in images.
Recurrent Networks (RNNs) and Transformers handle sequential information like text, speech, or time series.

The key principle is alignment: the model’s structure must align with the structure of the data.
A well-chosen architecture simplifies learning, while an unsuitable one forces the network to fight against the data instead of learning from it.

⸻

Hyperparameters and Configuration

Once the type of model is chosen, its configuration must be defined.
Hyperparameters determine how the network learns — how fast, how deep, and how flexible it becomes.

The learning rate (\eta) controls the size of the step taken during optimization.
If \eta is too large, the model overshoots the minimum; if too small, it learns too slowly or gets stuck.
The ideal configuration finds a balance between stability and speed.

The batch size, number of layers, number of neurons per layer, and activation functions also shape the model’s capacity.
These parameters are not learned automatically; they must be defined before training.

A simplified view of training iterations can be expressed as:

$$
\text{for each epoch: for each batch: update } w \leftarrow w - \eta \nabla L(w)
$$

where L(w) is the loss function and \nabla L(w) its gradient with respect to the weights.

Tuning hyperparameters is both an art and a science.
It often requires experimentation, intuition, and the use of tools like grid search, random search, or Bayesian optimization to explore the space of possibilities efficiently.

⸻

Model Validation

Validation ensures that what the model learns is truly general and not just memorized.
During training, the network continuously improves its accuracy on the training data, but that does not guarantee that it will perform well on new inputs.

To detect overfitting, the validation set is used as an external checkpoint.
When performance on the training data keeps improving while the validation score stagnates or worsens, the model is learning noise instead of structure.

A common strategy is k-fold cross-validation, where the dataset is divided into k parts. The model trains k times, each time leaving out one fold for validation and using the rest for training.
This approach provides a more reliable estimate of performance, reducing dependence on a single random split.

Mathematically, the average validation loss across all folds can be expressed as:

$$
L_{cv} = \frac{1}{k} \sum_{i=1}^{k} L_i
$$

where L_i is the loss obtained on the i^{th} validation fold.

Proper validation ensures that the model not only fits the data but understands it. It is the difference between memorization and generalization — between a fragile network and a trustworthy one.

⸻

Preparing, selecting, tuning, and validating are not separate steps but a continuous cycle.
Every adjustment in preprocessing may affect model behavior; every change in configuration may reveal new data issues.
Mastering this iterative process is what transforms a modeler into a true practitioner of neural networks.

# V. The Taxonomy of ANNs.

Artificial neural networks have grown into a large and diverse family of models.
Although they share the same foundation — neurons, weights, activations, and gradient-based learning — they differ in how they connect layers and transform information.
These structural variations define the taxonomy of ANNs, which serves as a conceptual map of the field.

Understanding this taxonomy is essential. It reveals why some models excel at vision, others at sequences, and others at compression or generation. It also highlights the evolution of neural architectures through decades of innovation.

⸻

Fundamental Structures

Every neural network can be described as a composition of transformations that convert inputs into outputs through successive layers:

$$
y = f_n(f_{n-1}(\dots f_1(x)))
$$

Each layer applies a weighted transformation followed by a nonlinear activation.
This simple principle — composition through layers — makes neural networks universal approximators, capable of representing almost any continuous function.

What distinguishes one network family from another is how each layer processes information.
Some connect every neuron to all inputs (dense networks), others focus on local spatial patterns (convolutions), temporal dependencies (recurrence), or global attention mechanisms (transformers).
This structural diversity defines the main branches of the ANN taxonomy.

⸻

1. Feedforward Networks

Feedforward Networks, also known as Multilayer Perceptrons (MLPs), are the earliest and simplest form of ANN.
Information flows in a single direction — from input to output — with no feedback loops or temporal dependencies.

Their general operation can be represented as:

$$
a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})
$$

where a^{(l)} denotes the activation of layer l, W^{(l)} and b^{(l)} are its weights and biases, and f is the activation function.

Feedforward networks introduced the concept of hidden layers, enabling the modeling of nonlinear relationships.
Their variants include the Single-Layer Perceptron, the Multilayer Perceptron (MLP), and the Radial Basis Function Network (RBFN), which uses Gaussian-like activations to capture local patterns in the input space.

Although they are the simplest members of the taxonomy, they laid the foundation for all subsequent architectures. Modern deep learning can be viewed as a natural extension of these basic structures — deeper, wider, and supported by more advanced optimization methods.

⸻

2. Convolutional Neural Networks (CNNs)

Convolutional Networks revolutionized how machines perceive visual and spatial data.
Instead of connecting every neuron to every pixel, CNNs use filters that slide over the input, learning local patterns like edges, textures, and shapes.

The convolution operation is defined as:

$$
s(t) = (x * w)(t) = \sum_{\tau} x(\tau) , w(t - \tau)
$$

Stacking multiple convolutional layers allows the network to build a hierarchy of features, where early layers capture simple visual components and deeper layers recognize complex structures.

Key CNN architectures mark milestones in this evolution.
LeNet-5 introduced the basic convolution–pooling sequence in 1998.
AlexNet (2012) demonstrated the power of deep CNNs on large-scale image datasets.
VGGNet simplified architectures with uniform 3×3 filters, while ResNet introduced residual connections to combat vanishing gradients.
Later models like DenseNet, Inception, and EfficientNet pushed efficiency, depth, and scalability even further.

Today, CNNs are applied beyond images — to audio spectrograms, time series, and even text embeddings — making them one of the most versatile structures in the ANN family.

⸻

3. Recurrent Neural Networks (RNNs)

Recurrent Networks were designed to handle sequential data, where order and context matter.
Unlike feedforward networks, they maintain an internal state that captures information from previous time steps, giving them a form of memory.

A recurrent neuron can be described mathematically as:

$$
h_t = f(W_h h_{t-1} + W_x x_t + b)
$$

where h_t is the hidden state at time t, x_t is the current input, and W_h, W_x, and b are learned parameters.

RNNs can process sequences of arbitrary length, making them effective for text, speech, and time-series analysis.
However, traditional RNNs struggle with long-term dependencies because of vanishing and exploding gradients.

To overcome this limitation, more advanced models were developed.
LSTM (Long Short-Term Memory) introduced gates to control the flow of information through time, while GRU (Gated Recurrent Unit) simplified the design while retaining efficiency.
Bidirectional RNNs read sequences both forward and backward, and Seq2Seq (Encoder–Decoder) architectures enabled machine translation and summarization.

These variants transformed how machines process sequences, forming the bridge between statistical modeling and modern language understanding.

⸻

4. Autoencoders

Autoencoders are neural networks that learn to represent data efficiently by reconstructing it.
They compress inputs into a latent representation (encoding) and then attempt to reconstruct the original data (decoding).

Their operation is defined by two mappings:

$$
z = f_{\text{enc}}(x), \quad \hat{x} = f_{\text{dec}}(z)
$$

where f_{\text{enc}} is the encoder, f_{\text{dec}} is the decoder, x is the input, and \hat{x} is the reconstructed output.

By minimizing the difference between x and \hat{x}, autoencoders learn to capture the underlying structure of data.
They have many variants, each serving a unique role:
	•	Denoising Autoencoder (DAE), which learns to reconstruct clean data from noisy inputs.
	•	Sparse Autoencoder, which enforces sparsity constraints to produce interpretable features.
	•	Convolutional Autoencoder (CAE), specialized for image reconstruction.
	•	Variational Autoencoder (VAE), which introduces probabilistic encoding and allows sampling from the latent space to generate new data.

Autoencoders connect the discriminative and generative sides of deep learning — they learn structure without supervision and form the foundation for more complex generative models.

⸻

5. Transformers

Transformers represent a paradigm shift in deep learning.
They replace recurrence with attention mechanisms that allow each element in a sequence to weigh its relationship to every other element.
This parallel processing enables faster training and captures long-range dependencies more effectively than RNNs.

The self-attention mechanism is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

where Q, K, and V are the query, key, and value matrices, and d_k is the dimension of the keys.

Transformers form the foundation of most modern models.
BERT (Bidirectional Encoder Representations from Transformers) improved language understanding by contextualizing words bidirectionally.
GPT (Generative Pretrained Transformer) specialized in text generation, while T5, XLNet, and PaLM extended these ideas with multitask and scaling innovations.
In computer vision, Vision Transformers (ViT) replaced convolutions with patch-based attention, and in speech, models like Whisper and wav2vec 2.0 brought similar advances.

Transformers unified the field by demonstrating that attention — not recurrence — could serve as the universal mechanism for learning across modalities.

⸻



7. Generative Networks


⸻

6. Hybrid and Advanced Architectures

Hybrid and advanced architectures combine ideas from multiple families, often blurring boundaries between them.
They are designed to handle complex, multimodal, or structured data, and to push the limits of representation learning.

Among these models are:
CNN–RNN Hybrids, which integrate spatial and temporal learning for video or sequential image data.
CNN–Transformer Hybrids, which merge local convolutional detail with global attention.
Graph Neural Networks (GNNs), which process relationships and dependencies across nodes and edges instead of sequences or grids.
Capsule Networks (CapsNets), which preserve hierarchical spatial relationships through dynamic routing.
Self-Organizing Maps (SOMs), inspired by biological maps, used for unsupervised clustering and visualization.
And Spiking Neural Networks (SNNs), which model real neuron firing patterns using discrete time spikes rather than continuous activations.

These architectures expand the boundaries of neural computation, connecting machine learning with neuroscience and complex systems.

⸻

This taxonomy is not static. New families continue to emerge, often combining existing ideas into hybrid paradigms that reflect the interdisciplinary nature of modern AI.
Understanding this structure — from perceptrons to transformers — provides the necessary foundation to explore the individual types of ANNs in depth.

# VI. The Types of Artificial Neural Networks

## Introduction and Purpose

Artificial Neural Networks (ANNs) form a universe of architectures that share a common mathematical DNA but express it in profoundly different ways. Each architecture was born to overcome a limitation of its predecessors — to capture a new dimension of learning that earlier models could not. The progression from simple feedforward layers to complex transformers and generative systems reflects both technological advances and deeper insights into how intelligence can be represented computationally.

This section explores those architectures in a coherent order, showing how each family adds a new structural principle: nonlinearity in Feedforward Networks, spatial locality in Convolutional Networks, temporal memory in Recurrent Networks, unsupervised compression in Autoencoders, attention and context in Transformers, creative synthesis in Generative Models, and integration in Hybrid and Advanced Architectures.

Our purpose is to move beyond isolated definitions and instead understand the logic that links them. Every network type will be analyzed through the same conceptual lenses so that patterns emerge naturally: how data flows, how knowledge is stored, and how learning evolves from shallow perception to deep representation.

By the end of this section, the reader should be able to see ANNs not as separate techniques but as expressions of a single, expanding idea — that learning systems grow by re-architecting the relationship between information, structure, and adaptation.

## Guiding Framework

To ensure coherence across all architectures, every neural model in this repository is described through a common analytical framework.
This unified structure allows fair comparison between algorithms, encourages conceptual clarity, and builds intuition layer by layer.

Rather than focusing on isolated code or formulas, each network type is examined as both a mathematical construct and a learning philosophy — revealing not only how it works, but why it matters.

Each model is therefore presented through the following ten analytical lenses:

1.	What is it?

A concise conceptual definition and a brief note on its historical or disciplinary origin.

2.	Why use it?

The problems, data structures, or scenarios where the model naturally excels.

3.	Intuition
The geometric, probabilistic, or algorithmic “mental picture” that explains how the model learns or transforms information.

4.	Mathematical foundation

The core principle behind its learning process — the estimation rule, optimization objective, or transformation equation — expressed in plain language and minimal notation.

5.	Training logic

A conceptual overview of how parameters are adjusted to minimize loss, maximize accuracy, or improve generalization during learning.

6.	Assumptions and limitations

The data conditions required for the model to perform well, and the contexts where it tends to fail or overfit.

7.	Key hyperparameters (conceptual view)

The main parameters that control model flexibility, depth, bias–variance balance, and capacity for generalization.

8.	Evaluation focus

The most relevant metrics, validation strategies, or diagnostic tools to assess performance — connecting back to the evaluation principles described in Section III.

9.	When to use / When not to use

Practical guidance on appropriate use cases and common misapplications.

10.	References

A short list of canonical academic sources (three) and authoritative web or video materials (two) for further reading and exploration.

This framework transforms the study of neural networks from a catalog of models into a comparative map of ideas — one where every architecture can be understood in relation to its predecessors and successors in the history of artificial intelligence.

## Transition to ANN Families

Every family of Artificial Neural Networks represents a response to a limitation of the previous one.
The evolution of these architectures mirrors the evolution of our understanding of learning itself: each generation introduced a new structural principle that allowed networks to perceive, remember, or create in increasingly sophisticated ways.

The earliest models, the Feedforward Networks, captured the essence of nonlinearity and layered abstraction — the idea that intelligence could emerge from the composition of simple transformations.
Then, Convolutional Neural Networks (CNNs) brought spatial awareness, learning to recognize patterns and hierarchies in images.
Soon after, Recurrent Neural Networks (RNNs) introduced temporal memory, allowing sequences and dependencies over time to be modeled directly.
The rise of Autoencoders shifted attention to unsupervised representation, compressing and reconstructing information without explicit labels.
Later, Transformers revolutionized the field with attention mechanisms, enabling parallel processing and long-range context understanding.
This paved the way for Generative Models, which moved from perception to creation, synthesizing data that resembles reality.
Finally, Hybrid and Advanced Architectures began to unify the strengths of all previous paradigms, forming the foundations of today’s most advanced AI systems.

Understanding this sequence is essential.
It reveals that progress in neural computation is not merely about building deeper models, but about expanding the dimensionality of understanding:
from static patterns → to spatial relationships → to temporal reasoning → to representation learning → to contextual attention → to creative synthesis → and ultimately integration.

In the next section, we will explore each family systematically.
The discussion begins with Feedforward Networks, the original blueprint from which all other neural architectures evolved.

## Canon of Models Covered (by Family)

The diversity of Artificial Neural Networks is vast. Dozens of architectures have emerged, each built upon the same foundational ideas of weighted connections and nonlinear transformations, but each designed to solve a distinct class of problems. To make this exploration coherent and focused, the models in this repository are organized by their shared structural principles rather than by chronology or application domain.

Each family reflects a conceptual leap — a shift in how information is represented and processed. The progression begins with Feedforward Networks, where information flows in one direction through layers of neurons. It then moves through families that introduce spatial locality (CNNs), temporal dependency (RNNs), unsupervised compression (Autoencoders), global attention (Transformers), creative synthesis (Generative Models), and structural integration (Hybrid and Advanced Architectures).

Within each family, there are many historical and modern variants. For instance, the Feedforward family includes the Single-Layer Perceptron (SLP), the Multilayer Perceptron (MLP), the Radial Basis Function Network (RBFN), the Extreme Learning Machine (ELM), and the Deep Feedforward Network (DFN).
To keep the discussion both rigorous and accessible, only the three most representative architectures from each family will be developed in depth — those that best illustrate the principles, mathematics, and evolution that define each paradigm.

This structure ensures that the reader can see how each innovation builds upon the last, transforming neural computation from a single linear neuron into the rich landscape of deep learning models we know today.

A. Feedforward Networks introduced nonlinearity and layered abstraction

Feedforward Networks mark the true beginning of artificial neural computation.
They embody the idea that intelligence can emerge from the layered composition of simple functions — a sequence of transformations that gradually turn raw input data into structured, meaningful representations.

Historically, this family was born in the mid-20th century, when researchers began to formalize the analogy between biological neurons and computational systems. The foundational moment came in 1943, with Warren McCulloch and Walter Pitts, who proposed the first mathematical model of a neuron capable of logical reasoning through weighted inputs and a binary activation threshold. Their work, “A Logical Calculus of the Ideas Immanent in Nervous Activity,” laid the conceptual groundwork for what would later become the perceptron.

In 1958, Frank Rosenblatt translated that theoretical neuron into a functioning machine: the Perceptron. Built with simple electrical circuits, it demonstrated that a computer could learn to classify patterns through experience rather than explicit programming. Rosenblatt’s perceptron was inspired by the brain’s visual cortex and aimed to mimic its capacity to detect and combine elementary features. The publication “The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain” became one of the most influential papers in early artificial intelligence.

The 1960s and 1970s, however, revealed the perceptron’s limitations. In 1969, Marvin Minsky and Seymour Papert published “Perceptrons,” showing that single-layer models could not solve non-linearly separable problems such as XOR. Their critique temporarily halted research on neural networks for nearly two decades. It was not until the 1980s that the field revived, following the rediscovery of the backpropagation algorithm by Rumelhart, Hinton, and Williams (1986). This algorithm allowed multi-layer networks to adjust internal weights efficiently, restoring confidence in connectionist learning and leading directly to the deep learning revolution decades later.

The Feedforward family thus represents the foundational principle of deep learning: hierarchical abstraction.
Each layer transforms its input into a more complex representation, moving from raw sensory data to abstract features and finally to a decision or prediction. Information flows strictly in one direction — from input to output — with no feedback loops. This unidirectional architecture allows simplicity, stability, and interpretability, making it the conceptual ancestor of all subsequent network designs.

The purpose of this family is not only computational but philosophical: to show that learning can emerge from the accumulation of simple nonlinear functions. It bridges biology, mathematics, and computer science in a single idea — that knowledge is structure, and structure can be learned.

In the following subsections, we will explore the three most representative models of this family:
the Single-Layer Perceptron (SLP), which formalized the concept of a neuron;
the Multilayer Perceptron (MLP), which introduced depth and backpropagation;
and the Radial Basis Function Network (RBFN), which redefined the notion of similarity in continuous spaces.

Main subtypes to cover:

1.	Single-Layer Perceptron (SLP) – the original neuron model by Rosenblatt.
2.	Multilayer Perceptron (MLP) – the standard deep feedforward network.
3.	Radial Basis Function Network (RBFN) – uses radial activation functions for localized responses.
4.	Extreme Learning Machine (ELM) – a fast random-weight feedforward alternative.
5.	Functional Link Neural Network (FLNN) – expands input space with nonlinear transformations.

In the following subsections, we will explore the three most representative models of this family:

- Single-Layer Perceptron (SLP), which formalized the concept of a neuron.
- Multilayer Perceptron (MLP), which introduced depth and backpropagation.
- Radial Basis Function Network (RBFN), which redefined the notion of similarity in continuous spaces.

1.	Single-Layer Perceptron (SLP) – the original neuron model by Rosenblatt.

What is it?
The Single-Layer Perceptron (SLP) is the earliest and simplest form of an Artificial Neural Network.
It consists of a single computational unit — or neuron — that takes multiple inputs, applies a set of learnable weights, adds a bias, and passes the result through an activation function to produce an output.

Formally, the perceptron can be described as:

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

where x_i are the input features, w_i are the corresponding weights, b is the bias, and f(\cdot) is a threshold activation function, typically the Heaviside step function.
This simple equation encodes the fundamental logic of neural computation — combining weighted evidence to make a decision.

The perceptron was introduced by Frank Rosenblatt in 1958 as a computational model inspired by the visual cortex. It was capable of learning from examples through weight adjustments, a radical idea at the time.
It marked the first successful attempt to build a machine that could generalize from data rather than follow explicit rules.

⸻

Why use it?
The SLP is a conceptual and educational cornerstone in machine learning.
Although too simple for modern applications, it remains essential for understanding how neural systems process and transform information.

It is primarily used for binary classification problems where classes are linearly separable.
Given labeled data, the perceptron learns a hyperplane that divides the input space into two regions, each corresponding to one class.
Because of its simplicity, it serves as a pedagogical model for explaining weight updates, activation functions, and the geometry of decision boundaries.

⸻

Intuition
At its core, the perceptron is a linear separator.
It projects data points into a space where a hyperplane — defined by its weights — splits one class from another.
Each weight w_i can be thought of as the importance assigned to feature x_i, while the bias b shifts the decision boundary.

Visually, in two dimensions, the perceptron learns a straight line that best divides the data into two categories.
The activation function acts as a binary switch: if the weighted sum exceeds a threshold, the output is one class; otherwise, it’s the other.

⸻

Mathematical Foundation
The perceptron learning algorithm aims to find a set of weights w_i that correctly classifies all training samples when the data are linearly separable.
The training proceeds iteratively, updating weights whenever a misclassification occurs.

If y_i is the true label (+1 or -1) and \hat{y}_i is the predicted output, the weight update rule is:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \eta (y_i - \hat{y}_i)\mathbf{x}_i
$$

where \eta is the learning rate that controls the step size of the update.
This rule moves the decision boundary closer to misclassified examples, gradually converging toward a correct separator if one exists.

⸻

Training Logic
Training the perceptron involves the following conceptual loop:
	1.	Initialize all weights and bias (often to small random values).
	2.	For each training example, compute the predicted output.
	3.	If the prediction is correct, do nothing.
	4.	If incorrect, adjust the weights using the update rule above.
	5.	Repeat until all examples are correctly classified or a maximum number of iterations is reached.

Because the perceptron only succeeds when the data are linearly separable, convergence is guaranteed in that case — as proven by Rosenblatt in 1962.

⸻

Assumptions and Limitations
The primary assumption of the perceptron is linear separability — that there exists a hyperplane dividing the data perfectly.
If this condition is not met (as in the XOR problem), the perceptron will fail to converge.

It also lacks any mechanism to represent nonlinear relationships, limiting its expressiveness to the simplest geometric boundaries.
Furthermore, it outputs only binary decisions, making it unsuitable for multi-class classification without extensions.

⸻

Key Hyperparameters (Conceptual View)
	•	Learning rate (η): Controls the magnitude of weight updates; too high leads to instability, too low slows convergence.
	•	Number of iterations: Defines how many times the dataset is processed.
	•	Activation function: Typically a binary step, though sigmoid or tanh can be used in variants.

The simplicity of these hyperparameters makes the SLP an ideal teaching tool for understanding optimization dynamics.

⸻

Evaluation Focus
The SLP is typically evaluated on its classification accuracy or convergence rate.
Given its deterministic behavior, performance is often analyzed geometrically — by visualizing the resulting decision boundary.
For linearly separable data, accuracy should reach 100%. For non-separable data, metrics such as the number of misclassifications per epoch or the margin size provide insight.

⸻

When to Use / When Not to Use
Use it when:
	•	The data are linearly separable or nearly so.
	•	The goal is to study or demonstrate the mechanics of learning and decision boundaries.
	•	Interpretability and simplicity are priorities.

Do not use it when:
	•	Relationships between features are nonlinear.
	•	The problem involves multiple classes, complex boundaries, or sequential dependencies.
	•	You require probabilistic outputs or deeper feature representations.

⸻

References
Canonical Papers
	1.	McCulloch, W. S., & Pitts, W. (1943). A Logical Calculus of the Ideas Immanent in Nervous Activity. Bulletin of Mathematical Biophysics.
	2.	Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. Psychological Review.
	3.	Minsky, M., & Papert, S. (1969). Perceptrons. MIT Press.

Web Resources
	1.	StatQuest – The Perceptron Clearly Explained: https://statquest.org/video/the-perceptron/￼
	2.	Scikit-learn Guide – Perceptron Model Overview: https://scikit-learn.org/stable/modules/linear_model.html#perceptron￼

----------

The Single-Layer Perceptron (SLP) laid the first stone in the architecture of artificial intelligence. It proved that machines could learn from data and that a simple weighted sum could separate classes through iterative adjustment. Yet, its simplicity was also its boundary. The SLP could only draw straight lines — or, more generally, hyperplanes — through the data. Whenever reality curved, it failed to follow.

Problems such as XOR, where classes are not linearly separable, exposed the perceptron’s geometric rigidity and inspired researchers to seek a model capable of bending those boundaries. The answer came in the form of depth — the introduction of hidden layers that could combine simple decisions into complex ones.

This evolution gave birth to the Multilayer Perceptron (MLP), the first true deep feedforward network, capable of approximating nonlinear functions and reshaping the field of machine learning.

----------

2.	Multilayer Perceptron (MLP) – the standard deep feedforward network.

What is it?
The Multilayer Perceptron (MLP) extends the perceptron’s original idea by introducing one or more hidden layers between the input and output.
Each hidden layer consists of neurons that apply nonlinear transformations, enabling the network to model complex, curved relationships that a single-layer perceptron cannot capture.

Formally, an MLP with one hidden layer can be expressed as:

$$
\hat{y} = f^{(2)}\left(W^{(2)} f^{(1)}(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)}\right)
$$

where f^{(1)} and f^{(2)} are activation functions, W^{(1)} and W^{(2)} are the weight matrices, and \mathbf{b}^{(1)}, \mathbf{b}^{(2)} are bias vectors.
This composition of layers enables the MLP to learn nonlinear decision boundaries by chaining multiple simple functions together.

The MLP became widely known after the rediscovery of the backpropagation algorithm by Rumelhart, Hinton, and Williams (1986), which made training deep networks computationally feasible.
This marked the beginning of modern connectionism, restoring neural networks to the center of machine learning research.

⸻

Why use it?
The MLP is the first universal function approximator.
It can represent any continuous mapping between inputs and outputs, given enough hidden units and proper training.
This flexibility makes it suitable for both regression and classification tasks across domains such as finance, medicine, natural language processing, and image analysis.

Its layered structure allows the model to automatically learn hierarchical representations — from raw features to abstract concepts — without the need for manual feature engineering.
This capacity for abstraction defines the very essence of deep learning.

⸻

Intuition
Each neuron in an MLP can be viewed as a filter that detects certain patterns in the input space.
The first layers capture simple structures, such as edges or directions, while deeper layers combine them into more complex representations.
The process resembles how the human brain builds understanding from low-level sensory input to high-level perception.

Intuitively, the MLP learns by bending the feature space — warping linear separations into nonlinear manifolds that fit the data’s true geometry.
The deeper the network, the more flexible the transformation.

⸻

Mathematical Foundation
The MLP relies on compositional function approximation:

$$
f(\mathbf{x}) = f^{(L)}(W^{(L)} f^{(L-1)}(\dots f^{(1)}(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)})) \dots ) + \mathbf{b}^{(L)})
$$

The Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991) states that a feedforward network with a single hidden layer and nonlinear activation can approximate any continuous function on compact subsets of \mathbb{R}^n, provided sufficient hidden units.

This theorem formalized what early researchers had observed: depth and nonlinearity grant networks the expressive power needed to learn virtually any pattern.

⸻

Training Logic
The learning process in an MLP is governed by backpropagation, an algorithm that computes the gradient of the loss function with respect to each weight by applying the chain rule of calculus backward through the network.

The typical steps are:
	1.	Forward pass: compute predictions from the current weights.
	2.	Loss calculation: measure the difference between predictions and true values.
	3.	Backward pass: propagate errors backward, layer by layer, computing gradients.
	4.	Weight update: adjust weights using an optimization rule such as Stochastic Gradient Descent (SGD).

The backpropagation equation for a weight w_{ij} connecting neuron i to j is:

$$
w_{ij}^{(t+1)} = w_{ij}^{(t)} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}
$$

where \eta is the learning rate and \mathcal{L} is the loss function.

⸻

Assumptions and Limitations
The MLP assumes that:
	•	The input data are independent and identically distributed (i.i.d.).
	•	The mapping between input and output is stationary (does not change over time).
	•	There is enough training data to learn stable weights.

However, MLPs face limitations:
	•	They are fully connected, which means every neuron in one layer connects to every neuron in the next. This makes them computationally heavy and prone to overfitting.
	•	They struggle with spatial or temporal dependencies because they ignore locality and sequence order.
	•	Their training can be slow and unstable without proper initialization or normalization.

These weaknesses eventually led to specialized architectures like CNNs and RNNs, which add inductive biases for structure and time.

⸻

Key Hyperparameters (Conceptual View)
	•	Number of hidden layers: Determines the network’s depth and representational capacity.
	•	Number of neurons per layer: Controls model complexity.
	•	Activation function: Common choices include ReLU, tanh, or sigmoid.
	•	Learning rate (η): Governs how quickly the model updates its weights.
	•	Batch size: Affects the stability and noise level of gradient estimates.
	•	Regularization terms: L1, L2, or dropout to prevent overfitting.

Each hyperparameter shapes the trade-off between underfitting, overfitting, and computational cost.

⸻

Evaluation Focus
Evaluation typically involves:
	•	Training and validation loss curves to assess convergence and overfitting.
	•	Accuracy, precision, recall, or F1-score for classification tasks.
	•	Mean squared error (MSE) or R² for regression problems.

Visualization of hidden-layer activations can also provide insight into how the model learns intermediate features.

⸻

When to Use / When Not to Use
Use it when:
	•	The relationship between input and output is nonlinear but lacks strong spatial or sequential structure.
	•	You need a flexible model for tabular or structured data.
	•	Interpretability can be secondary to predictive power.

Do not use it when:
	•	The dataset exhibits strong spatial correlation (images) or temporal order (sequences).
	•	Computational resources are limited, as fully connected networks scale poorly.
	•	The dataset is too small, leading to overfitting.

⸻

References

Canonical Papers

1.	Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning Representations by Back-Propagating Errors. Nature.
2.	Cybenko, G. (1989). Approximation by Superpositions of a Sigmoidal Function. Mathematics of Control, Signals and Systems.
3.	Hornik, K. (1991). Approximation Capabilities of Multilayer Feedforward Networks. Neural Networks.

Web Resources

1.	3Blue1Brown – What is Backpropagation Really Doing?: https://www.youtube.com/watch?v=tIeHLnjs5U8￼
2.	TensorFlow Guide – Building and Training a MLP: https://www.tensorflow.org/guide/keras/sequential_model￼


----------

The Multilayer Perceptron transformed neural networks from linear separators into universal approximators.
By adding depth and nonlinearity, it captured curved boundaries and complex patterns that were once unreachable.
Yet, its fully connected nature brought inefficiency — too many parameters, limited interpretability, and sensitivity to initialization.

In response, a new approach emerged: one that redefined similarity itself.
Instead of propagating inputs through layers of weights, it measured how close each example was to a set of learned prototypes in feature space.
This idea gave rise to the Radial Basis Function Network (RBFN) — a model that combines the geometry of distance with the flexibility of neural computation.

----------


3.	Radial Basis Function Network (RBFN) – uses radial activation functions for localized responses.

What is it?
The Radial Basis Function Network (RBFN) is a feedforward neural model that introduces a new way of interpreting learning: rather than adjusting weights globally across all inputs, it focuses on local similarity.
An RBF network measures how close each input is to a set of internal “prototype” points, using those distances to form the output.

Formally, an RBFN consists of three layers:
	1.	Input layer: receives the feature vector \mathbf{x}.
	2.	Hidden layer: applies a radial basis function to compute the similarity between \mathbf{x} and each prototype center \mathbf{c}_j.
	3.	Output layer: combines these activations linearly to produce the final prediction.

Mathematically:

$$
\hat{y} = \sum_{j=1}^{M} w_j , \phi(|\mathbf{x} - \mathbf{c}_j|) + b
$$

where \phi(\cdot) is a radial basis function, often the Gaussian kernel:

$$
\phi(r) = \exp\left(-\frac{r^2}{2\sigma^2}\right)
$$

Here, \mathbf{c}_j are the centers, \sigma controls the width (spread) of each kernel, and w_j are the linear output weights.
This formulation makes the RBFN conceptually close to kernel methods and Gaussian processes — models that learn by proximity rather than by transformation.

The architecture was first introduced by Broomhead and Lowe (1988) and Moody and Darken (1989) as an alternative to the multilayer perceptron, emphasizing local approximation over global fitting.

⸻

Why use it?
The RBFN excels in problems where the relationship between inputs and outputs is smooth, continuous, and locally structured.
Unlike the MLP, which modifies the entire decision surface whenever weights are updated, an RBFN adjusts only in regions near the relevant prototypes — making learning more interpretable and often more stable.

This local property makes RBFNs effective in:
	•	Function approximation with continuous targets.
	•	Time series prediction and control systems where local behavior matters.
	•	Pattern recognition tasks that rely on similarity or clustering intuition.

Because of their foundation on Gaussian functions, RBFNs naturally handle nonlinearity without requiring deep architectures.

⸻

Intuition
Intuitively, the RBFN transforms the input space into a landscape of localized responses.
Each hidden neuron activates when the input is close to its center, creating a “bubble” of influence in the feature space.
The network’s output is a smooth interpolation of these bubbles — the closer an input is to a prototype, the stronger its contribution.

This geometric interpretation links neural networks with the notion of distance-based reasoning in statistics and signal processing.
If MLPs bend the feature space globally, RBFNs tile it locally with overlapping regions of influence.

⸻

Mathematical Foundation
RBFNs approximate a target function f(\mathbf{x}) as a linear combination of radial basis functions centered at points \mathbf{c}_j:

$$
f(\mathbf{x}) = \sum_{j=1}^{M} w_j , \phi(|\mathbf{x} - \mathbf{c}_j|)
$$

Each center \mathbf{c}_j defines a neighborhood of influence determined by the kernel’s spread \sigma_j.
The choice of kernel determines smoothness — Gaussian kernels produce smooth, differentiable mappings, while multiquadric or thin-plate kernels yield sharper transitions.

Training involves finding the optimal centers, spreads, and output weights that minimize a loss function, typically mean squared error (MSE).

⸻

Training Logic
RBFNs are trained in two conceptual stages:
	1.	Center selection:
The centers \mathbf{c}_j can be determined by unsupervised methods like k-means clustering, random sampling, or gradient-based optimization.
	2.	Weight estimation:
Once the centers and spreads are fixed, the output weights w_j are obtained via linear regression:
$$
\mathbf{w} = (\Phi^T \Phi)^{-1} \Phi^T \mathbf{y}
$$
where \Phi is the matrix of basis function activations.
This separation of nonlinear transformation (hidden layer) and linear fitting (output layer) gives RBFNs a clear, interpretable structure.

⸻

Assumptions and Limitations
RBFNs assume that:
	•	The target function is locally smooth and can be modeled by overlapping radial functions.
	•	The number and placement of centers adequately cover the input space.

Their main limitations include:
	•	Scalability: performance degrades with high-dimensional data or many centers, since each adds computational cost.
	•	Choice of parameters: selecting centers, spreads (\sigma), and number of units requires careful tuning.
	•	Global generalization: unlike MLPs, RBFNs may fail to extrapolate beyond the regions covered by their centers.

⸻

Key Hyperparameters (Conceptual View)
	•	Number of centers (M): defines model capacity; more centers increase expressiveness but risk overfitting.
	•	Kernel type: Gaussian, multiquadric, or thin-plate spline; controls smoothness.
	•	Spread (σ): determines locality — small σ means narrow influence, large σ means overlapping responses.
	•	Regularization term: adds numerical stability when solving for output weights.

Balancing the number of centers and the spread width is crucial for effective learning.

⸻

Evaluation Focus
RBFNs are typically evaluated using regression-oriented metrics such as:
	•	Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) for continuous outputs.
	•	Accuracy or AUC when used for classification tasks after thresholding outputs.
Visualization of the learned centers and their spread can also provide interpretability — showing how the model partitions the input space.

⸻

When to Use / When Not to Use
Use it when:
	•	You need smooth interpolation or local function approximation.
	•	Data exhibit clear clusters or regions of influence.
	•	Interpretability of local behavior is important.

Do not use it when:
	•	Data are high-dimensional and not clusterable.
	•	The problem demands hierarchical feature abstraction.
	•	Training speed or memory efficiency is critical.

⸻

References
Canonical Papers
	1.	Broomhead, D. S., & Lowe, D. (1988). Multivariable Functional Interpolation and Adaptive Networks. Complex Systems.
	2.	Moody, J., & Darken, C. (1989). Fast Learning in Networks of Locally-Tuned Processing Units. Neural Computation.
	3.	Poggio, T., & Girosi, F. (1990). Networks for Approximation and Learning. Proceedings of the IEEE.

Web Resources
	1.	Towards Data Science – Understanding Radial Basis Function Networks: https://towardsdatascience.com/radial-basis-function-networks-explained￼
	2.	StatQuest – RBF Networks Clearly Explained: https://statquest.org/video/rbf-networks-explained/￼



----------

The Radial Basis Function Network (RBFN) closed the first era of feedforward architectures by shifting learning from global weight adjustment to local representation.
It demonstrated that neural computation could be both geometric and statistical — learning by proximity rather than by propagation.
Yet, RBFNs still treated all input dimensions equally, ignoring the spatial structure inherent in many types of data such as images, audio, or sensor maps.

The next leap came from mimicking the visual cortex more directly — teaching networks to recognize patterns through spatial hierarchies and local filters.
This gave rise to Convolutional Neural Networks (CNNs), which introduced the concept of spatial awareness into artificial intelligence.

----------


B. Convolutional Neural Networks (CNNs) introduced spatial awareness

CNNs are designed for spatial data — mainly images, videos, or structured grids — and evolved through increasingly deep and efficient architectures.

Main subtypes to cover:

1.	LeNet-5 – the first successful CNN for handwritten digit recognition.
2.	AlexNet – introduced deep CNNs with ReLU activations and GPUs.
3.	VGGNet – uniform convolution blocks, simplicity through depth.
4.	GoogLeNet (Inception) – multi-scale convolutions for efficiency.
5.	ResNet – residual connections to enable very deep networks.
6.	DenseNet – dense connectivity between layers for gradient flow.
7.	MobileNet – lightweight CNN optimized for mobile devices.
8.	EfficientNet – compound scaling of depth, width, and resolution.
9.	Vision Transformers (ViT) – transformer-based vision model that bridges CNN and attention paradigms (transitional model).

C. Recurrent Neural Networks (RNNs) introduced temporal memory

RNNs handle sequential or time-dependent data, learning from context and order.

Main subtypes to cover:
1.	Simple RNN (Elman or Jordan Network) – base sequential model.
2.	Long Short-Term Memory (LSTM) – introduces gating to preserve long-term dependencies.
3.	Gated Recurrent Unit (GRU) – a simplified LSTM variant.
4.	Bidirectional RNN (BiRNN) – processes data forward and backward.
5.	Encoder–Decoder (Seq2Seq) – foundation for translation and summarization.
6.	Attention-based RNNs – transitional form leading to Transformers.
7.	Echo State Networks (ESN) – reservoir computing variant.

D. Autoencoders introduced unsupervised compression

Autoencoders learn to reconstruct input data, enabling dimensionality reduction, denoising, and generative representation learning.

Main subtypes to cover:

1.	Basic Autoencoder (AE) – classical encoder–decoder design.
2.	Sparse Autoencoder – encourages sparse activation for interpretability.
3.	Denoising Autoencoder (DAE) – reconstructs clean data from noise.
4.	Contractive Autoencoder (CAE) – penalizes sensitivity in latent space.
5.	Convolutional Autoencoder (ConvAE) – adapted for images.
6.	Variational Autoencoder (VAE) – probabilistic latent variables for generation.
7.	Adversarial Autoencoder (AAE) – merges AE and GAN principles.
8.	Stacked Autoencoder (SAE) – multiple AEs combined for deep representation.
  
E. Transformers introduced attention and parallel sequence processing

Transformers replaced recurrence with attention, allowing scalable learning across sequences, text, and vision.

Main subtypes to cover:

1.	Original Transformer (Vaswani et al., 2017) – encoder–decoder with self-attention.
2.	BERT (Bidirectional Encoder Representations) – contextual understanding of language.
3.	GPT Series (Generative Pretrained Transformer) – autoregressive text generation (GPT-1 to GPT-4).
4.	T5 (Text-to-Text Transfer Transformer) – unified text framework for all NLP tasks.
5.	XLNet – permutation-based training to improve bidirectional context.
6.	ALBERT / RoBERTa / DistilBERT – efficiency and fine-tuning variants.
7.	Vision Transformer (ViT) – image classification via patch embeddings.
8.	Whisper – speech recognition with Transformer architecture.
9.	Multimodal Transformers (CLIP, Flamingo, Gemini) – integrate text, image, and audio modalities.


F. Generative Models introduced creativity and data synthesis

Generative models learn data distributions and create new, realistic samples.
They mark the shift from recognition to imagination in neural computation.

Main subtypes to cover:

1.	Generative Adversarial Network (GAN) – adversarial generator–discriminator setup.
2.	Deep Convolutional GAN (DCGAN) – CNN-based GAN architecture.
3.	Conditional GAN (cGAN) – conditioning generation on labels or context.
4.	CycleGAN – unpaired image-to-image translation.
5.	StyleGAN (1, 2, 3) – high-fidelity image generation.
6.	Variational Autoencoder (VAE) – probabilistic generation model (shared with D family).
7.	Diffusion Models (DDPM, Stable Diffusion) – iterative noise-to-image transformation.
8.	Flow-based Models (RealNVP, Glow) – invertible transformation models.
9.	Energy-based Models (EBMs) – probability modeling via energy functions.


G. Hybrid and Advanced Architectures unified previous paradigms

These models blend ideas from multiple families or extend the concept of “network” beyond fixed topologies.

Main subtypes to cover:

1.	CNN–RNN Hybrids – spatial + temporal learning (video, medical imaging).
2.	CNN–Transformer Hybrids – combine local feature extraction with global attention.
3.	Graph Neural Networks (GNNs) – reasoning over graph-structured data.
4.	Capsule Networks (CapsNets) – hierarchical spatial relationships.
5.	Self-Organizing Maps (SOMs) – topological mapping of features.
6.	Spiking Neural Networks (SNNs) – biologically inspired temporal spikes.
7.	Neural ODEs – continuous-time deep learning.
8.	Liquid Neural Networks (LNNs) – adaptive, dynamic neuron models.
9.	Neural Radiance Fields (NeRFs) – 3D scene representation and rendering.
10.	Neural Architecture Search (NAS) – automated design of neural networks.

## Summary of NNA family

# VII. Applications of Artificial Neural Networks

Bridges theory with practice.
Here we show real-world implementations — notebooks or code examples — illustrating how different architectures solve specific problems.

Subsections (examples):

1.	Image Classification and Object Detection
2.	Natural Language Processing and Sentiment Analysis
3.	Time Series Forecasting
4.	Anomaly Detection and Predictive Maintenance
5.	Generative Art, Image-to-Image Translation, and Text Generation

This section would directly connect to your practical folders (04_Aplicaciones and 03_Implementaciones).

# VIII. Annex and References

Final documentation section — complementary materials and supporting references.

Subsections (examples):

1.	Formulas in LaTeX — all mathematical expressions organized for quick reference.
2.	Glossary of Terms — clear definitions of technical terminology.
3.	Personal Notes — reflections, insights, or observations about learning and experimentation.
4.	Reference Materials — key textbooks, scientific papers, and reliable web sources.
5.	License and Academic Purpose — short statement about the open and educational intent of the repository.
