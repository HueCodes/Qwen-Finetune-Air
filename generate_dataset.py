#!/usr/bin/env python3
"""
Generate a comprehensive, high-quality training dataset
Focus on diverse ML/AI topics with detailed, accurate answers
"""
import json
import random

# ==============================================================================
# COMPREHENSIVE Q&A DATABASE
# ==============================================================================

# 1. MACHINE LEARNING FUNDAMENTALS (50+ questions)
ml_fundamentals = [
    ("What is machine learning?", "Machine learning is a branch of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions based on those patterns."),
    ("Explain supervised learning.", "Supervised learning is a machine learning paradigm where models learn from labeled training data. The algorithm learns a mapping from inputs to outputs using examples, then applies this learned mapping to make predictions on new, unseen data. Common tasks include classification and regression."),
    ("What is unsupervised learning?", "Unsupervised learning discovers hidden patterns in unlabeled data without predefined categories. The algorithm explores data structure through techniques like clustering, dimensionality reduction, and density estimation, revealing natural groupings or relationships."),
    ("Describe reinforcement learning.", "Reinforcement learning trains agents to make sequential decisions by interacting with an environment. The agent receives rewards or penalties for actions and learns to maximize cumulative reward over time through trial and error."),
    ("What is the bias-variance tradeoff?", "The bias-variance tradeoff describes the balance between a model's ability to fit training data (low bias) and generalize to new data (low variance). High bias leads to underfitting, high variance to overfitting. Optimal models balance both."),
    ("Explain the difference between classification and regression.", "Classification predicts discrete categories or labels (e.g., cat vs dog), while regression predicts continuous numerical values (e.g., temperature). Both are supervised learning tasks but differ in output type and evaluation metrics."),
    ("What is a training set?", "A training set is the portion of data used to train a machine learning model. The model learns patterns, relationships, and parameters from this data. It's typically the largest split, often 60-80% of total data."),
    ("What is a validation set?", "A validation set is data held out from training to tune hyperparameters and make model selection decisions. It provides unbiased evaluation during development without touching the test set, preventing overfitting to test data."),
    ("What is a test set?", "A test set is data completely held out from training and validation, used only for final model evaluation. It provides an unbiased estimate of how the model will perform on new, unseen data in production."),
    ("What is overfitting?", "Overfitting occurs when a model learns training data too well, including noise and outliers. The model memorizes rather than generalizes, achieving high training accuracy but poor performance on new data. Signs include large gap between train and validation performance."),
    ("What is underfitting?", "Underfitting happens when a model is too simple to capture underlying data patterns. It performs poorly on both training and test data because it lacks the capacity to learn the true relationship between inputs and outputs."),
    ("Explain the curse of dimensionality.", "The curse of dimensionality refers to phenomena that arise when working with high-dimensional data. As dimensions increase, data becomes sparse, distances lose meaning, and exponentially more samples are needed to maintain density and statistical significance."),
    ("What is feature engineering?", "Feature engineering is the process of creating new features or transforming existing ones to improve model performance. It involves domain knowledge to extract meaningful representations that make patterns more apparent to learning algorithms."),
    ("What is feature selection?", "Feature selection identifies and retains only the most relevant features for a task, removing irrelevant or redundant ones. This reduces dimensionality, improves model interpretability, decreases training time, and can prevent overfitting."),
    ("Explain cross-validation.", "Cross-validation assesses model performance by partitioning data into k folds, iteratively training on k-1 folds and validating on the remaining fold. Results are averaged across all folds, providing robust performance estimates and reducing variance."),
    ("What is k-fold cross-validation?", "K-fold cross-validation divides data into k equal parts (folds). The model trains k times, each time using a different fold for validation and the remaining k-1 for training. Common choices are k=5 or k=10."),
    ("What is stratified cross-validation?", "Stratified cross-validation ensures each fold has approximately the same class distribution as the original dataset. This is crucial for imbalanced datasets to ensure representative training and validation sets in each fold."),
    ("What is leave-one-out cross-validation?", "Leave-one-out cross-validation (LOOCV) is an extreme case of k-fold CV where k equals the dataset size. Each iteration uses one sample for validation and all others for training. It's computationally expensive but provides maximum training data per iteration."),
    ("Explain ensemble learning.", "Ensemble learning combines multiple models to produce better predictions than any individual model. By aggregating diverse models' outputs, ensembles reduce variance, increase robustness, and often achieve superior performance."),
    ("What is bagging?", "Bagging (Bootstrap Aggregating) trains multiple models on different random samples of training data (with replacement) and averages their predictions. It reduces variance and prevents overfitting, with Random Forests being a popular example."),
    ("What is boosting?", "Boosting sequentially trains weak learners, each focusing on examples the previous models struggled with. It combines these weak learners into a strong ensemble that often achieves excellent performance. Examples include AdaBoost, Gradient Boosting, and XGBoost."),
    ("What is stacking?", "Stacking (Stacked Generalization) trains a meta-model to combine predictions from multiple base models. The meta-model learns optimal weights or combinations of base predictions, often outperforming simple averaging."),
]

# 2. NEURAL NETWORKS & DEEP LEARNING (50+ questions)
neural_networks = [
    ("What is a neural network?", "A neural network is a computational model inspired by biological neurons, consisting of interconnected nodes (artificial neurons) organized in layers. It learns to map inputs to outputs through adjustable weights that are optimized during training."),
    ("Explain the perceptron.", "The perceptron is the simplest neural network unit, computing a weighted sum of inputs plus a bias, then applying a step activation function. While limited to linear decision boundaries, it forms the foundation of modern neural networks."),
    ("What is a multilayer perceptron?", "A multilayer perceptron (MLP) is a feedforward neural network with one or more hidden layers between input and output. Hidden layers with non-linear activations enable learning complex, non-linear relationships between inputs and outputs."),
    ("What is backpropagation?", "Backpropagation is the algorithm for training neural networks. It computes gradients of the loss function with respect to each weight by propagating errors backward through the network using the chain rule, enabling gradient descent optimization."),
    ("What is an activation function?", "An activation function introduces non-linearity into neural networks, enabling them to learn complex patterns. It determines whether and how strongly a neuron fires based on its input. Examples include ReLU, sigmoid, and tanh."),
    ("Explain the ReLU activation function.", "ReLU (Rectified Linear Unit) outputs max(0, x), zero for negative inputs and identity for positive. It's computationally efficient, mitigates vanishing gradients, and induces sparsity. It's the most popular activation for hidden layers in deep networks."),
    ("What is the sigmoid activation function?", "The sigmoid function maps inputs to (0,1) using the formula 1/(1+e^-x). It's useful for binary classification output layers and was historically popular for hidden layers, though it suffers from vanishing gradients."),
    ("What is the tanh activation function?", "The tanh (hyperbolic tangent) function maps inputs to (-1,1). It's zero-centered, making optimization easier than sigmoid, but also suffers from vanishing gradients. It's sometimes used in RNNs and specific architectures."),
    ("What is the softmax function?", "Softmax converts a vector of values into a probability distribution that sums to 1. It's used in multi-class classification output layers, exponentiating each element and normalizing by the sum."),
    ("What is a loss function?", "A loss function (or cost function) quantifies the difference between model predictions and true labels. It provides a differentiable objective that optimization algorithms minimize during training to improve model performance."),
    ("Explain cross-entropy loss.", "Cross-entropy loss measures the difference between predicted probability distributions and true labels. For classification, it's the negative log probability of the correct class. It's the standard loss for classification tasks."),
    ("What is mean squared error?", "Mean squared error (MSE) averages the squared differences between predictions and targets. It's the standard loss function for regression, penalizing large errors more heavily than small ones due to squaring."),
    ("What is mean absolute error?", "Mean absolute error (MAE) averages the absolute differences between predictions and targets. Unlike MSE, it treats all errors equally and is more robust to outliers, making it useful when large errors should not be overly penalized."),
    ("What is gradient descent?", "Gradient descent is an iterative optimization algorithm that adjusts parameters in the direction of steepest descent of the loss function. It uses gradients (derivatives) to determine update direction and magnitude, moving toward a local minimum."),
    ("Explain stochastic gradient descent.", "Stochastic gradient descent (SGD) updates parameters using gradients computed from single samples or small mini-batches rather than the entire dataset. This introduces noise but enables faster iterations, better generalization, and escape from shallow local minima."),
    ("What is mini-batch gradient descent?", "Mini-batch gradient descent computes gradients on small batches of samples (e.g., 32-512), balancing between full-batch stability and stochastic speed. It's the most common variant used in practice, offering efficient parallelization and stable convergence."),
    ("What is the learning rate?", "The learning rate controls the step size in gradient descent optimization. Too high causes divergence or oscillation around minima; too low results in slow convergence or getting stuck. It's one of the most critical hyperparameters requiring careful tuning."),
    ("What is momentum in optimization?", "Momentum accelerates gradient descent by accumulating a velocity vector in directions of persistent gradient reduction. It helps overcome local minima, dampens oscillations, and speeds convergence, especially in ravines or plateaus."),
    ("What is the Adam optimizer?", "Adam (Adaptive Moment Estimation) combines momentum and adaptive learning rates per parameter. It maintains exponential moving averages of gradients and squared gradients, adapting learning rates based on first and second moment estimates for efficient optimization."),
    ("What is the vanishing gradient problem?", "The vanishing gradient problem occurs when gradients become exponentially small during backpropagation through many layers. This happens with saturating activations (sigmoid/tanh) and prevents deep network training as early layers receive tiny updates."),
    ("What is the exploding gradient problem?", "The exploding gradient problem occurs when gradients grow exponentially large during backpropagation, causing numerical instability. It's common in RNNs with long sequences and can be mitigated with gradient clipping, careful initialization, or architecture changes."),
    ("What is batch normalization?", "Batch normalization normalizes layer inputs across mini-batches to have zero mean and unit variance. This reduces internal covariate shift, stabilizes training, allows higher learning rates, and acts as a regularizer, significantly accelerating deep network training."),
    ("What is layer normalization?", "Layer normalization normalizes across features for each sample independently, unlike batch norm which normalizes across the batch. It's useful for RNNs and transformers where batch sizes vary or batch statistics are unreliable."),
    ("What is dropout?", "Dropout randomly deactivates neurons during training with probability p (typically 0.5). This prevents co-adaptation of features, creates an ensemble effect by training different sub-networks, and serves as powerful regularization for deep networks."),
    ("What is weight initialization?", "Weight initialization sets initial parameter values before training. Poor initialization can cause vanishing/exploding gradients or slow convergence. Modern methods like Xavier and He initialization account for layer sizes and activations for stable training."),
    ("What is Xavier initialization?", "Xavier (Glorot) initialization sets weights from a distribution with variance 1/n_in (or 2/(n_in+n_out)), where n_in and n_out are input and output dimensions. It maintains activation and gradient variance across layers for stable training with tanh/sigmoid."),
    ("What is He initialization?", "He initialization uses variance 2/n_in, doubling Xavier's variance. It's designed for ReLU activations which kill half the gradients. This compensation maintains stable gradient flow through deep ReLU networks."),
]

# 3. COMPUTER VISION (30+ questions)
computer_vision = [
    ("What is a convolutional neural network?", "A CNN is a neural network specialized for processing grid-structured data like images. It uses convolutional layers to detect local patterns through learnable filters, pooling layers to reduce spatial dimensions, and fully connected layers for classification."),
    ("Explain convolutional layers.", "Convolutional layers apply learnable filters (kernels) to input feature maps through convolution operations. Each filter detects specific patterns (edges, textures, objects) and produces feature maps showing where those patterns occur spatially."),
    ("What is a convolution operation?", "Convolution slides a small filter (kernel) across an input, computing element-wise multiplication and summing results at each position. This creates a feature map indicating filter response strength across spatial locations, detecting local patterns."),
    ("What is a pooling layer?", "Pooling layers downsample feature maps by combining neighboring values, typically using max or average operations. This reduces spatial dimensions, computational cost, and provides translation invariance while retaining important features."),
    ("What is max pooling?", "Max pooling selects the maximum value within each pooling window, retaining the strongest activation. It provides translation invariance, reduces dimensions, and helps the network focus on whether features are present rather than their exact location."),
    ("What is average pooling?", "Average pooling computes the mean of values within each pooling window. It provides smoother downsampling than max pooling and is sometimes used in network final layers to aggregate spatial information."),
    ("What are feature maps?", "Feature maps are the outputs of convolutional or pooling layers, representing the presence and strength of learned features at different spatial locations. Early layers detect simple features (edges), while deeper layers detect complex patterns (objects)."),
    ("What is a filter/kernel?", "A filter (or kernel) is a small matrix of learnable weights that slides across inputs in convolutional layers. It detects specific patterns through element-wise multiplication and summation, producing high responses where patterns match."),
    ("Explain stride in convolution.", "Stride determines the step size when sliding filters across inputs. Stride 1 moves one pixel at a time; stride 2 skips every other position. Larger strides reduce output dimensions and computational cost but may miss fine details."),
    ("What is padding in CNNs?", "Padding adds border pixels (usually zeros) around inputs before convolution. This preserves spatial dimensions, allows filters to process edge pixels effectively, and prevents rapid shrinking of feature maps through network depth."),
    ("What is transfer learning in computer vision?", "Transfer learning uses models pre-trained on large datasets (like ImageNet) as starting points for new tasks. The network has already learned useful features, requiring less data and training time for downstream tasks through fine-tuning."),
    ("What is data augmentation in images?", "Data augmentation artificially increases training data diversity by applying transformations like rotations, flips, crops, color jitter, and noise. This improves model robustness, reduces overfitting, and teaches invariance to common variations."),
    ("Explain the VGG architecture.", "VGG uses very deep networks (16-19 layers) with small 3×3 convolutional filters stacked repeatedly. This simple, uniform architecture demonstrates that network depth significantly impacts performance, though it's computationally expensive."),
    ("What is ResNet?", "ResNet introduces skip connections (residual connections) that add layer inputs directly to outputs. This enables training very deep networks (50-152+ layers) by allowing gradients to flow directly through the network, solving degradation problems."),
    ("What are residual connections?", "Residual (skip) connections add layer inputs to outputs, learning residual mappings F(x) instead of desired mappings H(x) = F(x) + x. This enables training very deep networks by providing gradient highways and making learning easier."),
    ("What is object detection?", "Object detection identifies and localizes multiple objects in images, predicting both class labels and bounding boxes. It's more challenging than classification as it requires determining what objects exist and where they are located."),
    ("What is semantic segmentation?", "Semantic segmentation assigns a class label to every pixel in an image, creating dense predictions. It's used for tasks like medical image analysis, autonomous driving, and scene understanding where precise boundaries matter."),
    ("What is instance segmentation?", "Instance segmentation distinguishes between different instances of the same class, combining object detection and semantic segmentation. It assigns unique labels to each object instance, useful for counting or tracking individual objects."),
]

# 4. NATURAL LANGUAGE PROCESSING (30+ questions)
nlp = [
    ("What is natural language processing?", "Natural language processing (NLP) enables computers to understand, interpret, and generate human language. It combines linguistics, computer science, and machine learning to process text and speech for tasks like translation, sentiment analysis, and question answering."),
    ("What is tokenization?", "Tokenization breaks text into smaller units (tokens) such as words, subwords, or characters. It's the first step in NLP pipelines, converting raw text into structured units that models can process numerically."),
    ("What is word embedding?", "Word embeddings represent words as dense, low-dimensional vectors that capture semantic relationships. Words with similar meanings have similar vectors, enabling models to understand semantic similarity and perform arithmetic with meaning (king - man + woman ≈ queen)."),
    ("Explain Word2Vec.", "Word2Vec learns word embeddings by predicting words from context (CBOW) or context from words (Skip-gram). It trains a shallow neural network on large corpora, producing embeddings where semantically similar words cluster together in vector space."),
    ("What is GloVe?", "GloVe (Global Vectors) learns embeddings by factorizing word co-occurrence matrices. It captures both local context and global corpus statistics, combining benefits of matrix factorization and local context window methods like Word2Vec."),
    ("What is a recurrent neural network?", "RNNs process sequential data by maintaining hidden states that carry information across time steps. Each step's output depends on current input and previous hidden state, enabling the network to capture temporal dependencies in sequences."),
    ("Explain LSTM networks.", "Long Short-Term Memory (LSTM) networks address RNN's vanishing gradient problem using gating mechanisms (input, forget, output gates) and memory cells. They can learn long-range dependencies by selectively remembering or forgetting information."),
    ("What is GRU?", "Gated Recurrent Units (GRU) simplify LSTMs by combining forget and input gates into an update gate and merging cell and hidden states. They achieve similar performance to LSTMs with fewer parameters and faster training."),
    ("What is the attention mechanism?", "Attention allows models to focus on relevant input parts when processing each output. It computes weighted combinations of inputs based on learned relevance scores, enabling models to handle variable-length sequences and capture long-range dependencies effectively."),
    ("What is self-attention?", "Self-attention computes attention within a single sequence, relating different positions to each other. Each element attends to all elements (including itself), learning which parts are relevant for representing each position in context."),
    ("What is the transformer architecture?", "Transformers process sequences using self-attention mechanisms instead of recurrence. They enable parallel computation, capture long-range dependencies efficiently, and form the foundation of modern NLP models like BERT, GPT, and T5."),
    ("Explain the encoder-decoder architecture.", "Encoder-decoder architectures consist of an encoder that processes input into representations and a decoder that generates output. They're used for sequence-to-sequence tasks like translation, where input and output lengths differ."),
    ("What is BERT?", "BERT (Bidirectional Encoder Representations from Transformers) pre-trains bidirectional transformers on large corpora using masked language modeling and next sentence prediction. It captures deep bidirectional context and achieves state-of-the-art results on many NLP tasks."),
    ("What is GPT?", "GPT (Generative Pre-trained Transformer) is an autoregressive language model that predicts next tokens given previous context. It's pre-trained on large text corpora and can be fine-tuned for various tasks or used for few-shot learning."),
    ("What is masked language modeling?", "Masked language modeling randomly masks tokens in text and trains models to predict masked tokens from context. BERT uses this self-supervised objective to learn bidirectional representations from unlabeled text at scale."),
    ("What is next sentence prediction?", "Next sentence prediction is a BERT pre-training task that determines if two sentences are consecutive in the original text. It helps the model learn sentence-level relationships useful for tasks like question answering and natural language inference."),
    ("What is named entity recognition?", "Named Entity Recognition (NER) identifies and classifies named entities (people, organizations, locations, dates) in text. It's a fundamental NLP task used in information extraction, question answering, and knowledge graph construction."),
    ("What is sentiment analysis?", "Sentiment analysis determines the emotional tone or opinion expressed in text (positive, negative, neutral). It's widely used for analyzing customer feedback, social media monitoring, and market research."),
    ("What is machine translation?", "Machine translation automatically translates text from one language to another. Modern neural machine translation uses encoder-decoder architectures with attention, achieving near-human performance for some language pairs."),
    ("What is question answering?", "Question answering systems take a question and context (or knowledge base) and generate or extract an answer. It combines reading comprehension, information retrieval, and natural language understanding."),
]

# 5. LoRA & FINE-TUNING (40+ questions)
lora_finetuning = [
    ("What is fine-tuning in machine learning?", "Fine-tuning adapts a pre-trained model to a specific task by continuing training on task-specific data. It leverages knowledge learned from large datasets, requiring less data and computation than training from scratch while achieving better performance."),
    ("Explain transfer learning.", "Transfer learning applies knowledge from models trained on one task to improve performance on different but related tasks. Pre-trained models capture useful features and representations that transfer across domains, enabling efficient learning with limited data."),
    ("What is catastrophic forgetting?", "Catastrophic forgetting occurs when a neural network abruptly loses previously learned knowledge upon learning new information. It's a major challenge in continual learning, as updating weights for new tasks can overwrite weights important for old tasks."),
    ("What is parameter-efficient fine-tuning?", "Parameter-efficient fine-tuning adapts large pre-trained models while updating only a small fraction of parameters. Methods like LoRA, adapters, and prefix tuning achieve comparable performance to full fine-tuning with drastically reduced memory and computational requirements."),
    ("What is LoRA?", "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that decomposes weight updates into low-rank matrices. Instead of updating weight W directly, it learns W + AB where A and B are much smaller, reducing trainable parameters by orders of magnitude."),
    ("How does LoRA work mathematically?", "LoRA represents weight updates as a product of two low-rank matrices: ΔW = AB, where A is (d×r) and B is (r×d), with rank r << d. During inference, W' = W₀ + αAB, where W₀ is frozen pre-trained weights and α is a scaling factor."),
    ("Why is LoRA effective?", "LoRA is effective because weight updates during fine-tuning often have low intrinsic dimensionality. By constraining updates to low-rank subspaces, it captures essential adaptations while dramatically reducing parameters. This also acts as regularization, preventing overfitting."),
    ("What is the rank in LoRA?", "The rank (r) in LoRA determines the dimensionality of the low-rank decomposition. Lower rank means fewer parameters and faster training but less expressiveness. Higher rank provides more capacity but increases cost. Typical values range from 4 to 64."),
    ("What is LoRA alpha?", "LoRA alpha (α) is a scaling hyperparameter that controls the magnitude of LoRA updates relative to original weights. The effective learning rate for LoRA parameters is scaled by α/r, where r is rank. Common practice sets α = 2r."),
    ("What modules should be targeted with LoRA?", "LoRA typically targets attention weight matrices (Query, Key, Value, Output projections) as they contain most parameters and are most important for adaptation. Some implementations also target feed-forward network layers for additional capacity."),
    ("What is QLoRA?", "QLoRA (Quantized LoRA) combines LoRA with 4-bit quantization of base model weights. It enables fine-tuning of very large models on consumer GPUs by drastically reducing memory requirements while maintaining competitive performance through careful quantization and LoRA adaptation."),
    ("Explain adapter layers.", "Adapter layers are small bottleneck modules inserted into pre-trained models. They consist of down-projection, non-linearity, and up-projection layers. Only adapter parameters are trained while original weights stay frozen, enabling efficient multi-task learning."),
    ("What is prefix tuning?", "Prefix tuning prepends learnable continuous vectors (virtual tokens) to input sequences at each layer. These prefix parameters are optimized while the entire pre-trained model stays frozen, guiding model behavior for specific tasks with minimal parameters."),
    ("What is prompt tuning?", "Prompt tuning optimizes continuous prompt embeddings prepended to inputs while keeping the entire model frozen. It's an extremely parameter-efficient method requiring only a small prompt matrix, effective for large models where even LoRA might be expensive."),
    ("What is soft prompting?", "Soft prompting learns continuous prompt embeddings in the model's embedding space rather than discrete text prompts. Unlike hard prompts (actual text), soft prompts can occupy regions of embedding space not reachable by vocabulary tokens, providing more flexibility."),
    ("What are the advantages of LoRA over full fine-tuning?", "LoRA requires 10-100x fewer trainable parameters, drastically reducing memory usage and training time. It enables fine-tuning large models on consumer hardware, allows storing multiple task-specific adapters cheaply, and provides implicit regularization against overfitting."),
    ("What are the disadvantages of LoRA?", "LoRA adds inference latency if not merged with base weights, requires careful hyperparameter tuning (rank, alpha), and may underperform full fine-tuning on tasks requiring substantial behavior changes. The optimal rank varies across tasks and models."),
    ("How do you merge LoRA weights?", "LoRA weights can be merged into base model weights after training: W_merged = W_base + α × A × B. This eliminates inference overhead from adapter modules while preserving task-specific adaptations in a single model that can be deployed normally."),
    ("Can multiple LoRA adapters be used together?", "Yes, multiple LoRA adapters can be combined for different tasks or aspects. They can be weighted and summed: W = W_base + α₁A₁B₁ + α₂A₂B₂. This enables multi-task models, mixing capabilities, or compositional task solving."),
    ("What is the intrinsic dimensionality hypothesis?", "The intrinsic dimensionality hypothesis posits that successful model updates lie in a much lower-dimensional subspace than the full parameter space. This explains why low-rank methods like LoRA work well: the essential adaptations fit in small subspaces."),
]

# 6. TRAINING DYNAMICS & OPTIMIZATION (30+ questions)
training_optimization = [
    ("What is a hyperparameter?", "Hyperparameters are configuration settings that control the learning process but aren't learned from data. Examples include learning rate, batch size, number of layers, and regularization strength. They must be set before training and tuned for optimal performance."),
    ("What is hyperparameter tuning?", "Hyperparameter tuning systematically searches for optimal hyperparameter values. Methods include grid search (exhaustive), random search (efficient), and Bayesian optimization (smart). Good tuning can dramatically improve model performance."),
    ("What is grid search?", "Grid search exhaustively evaluates model performance across all combinations of predefined hyperparameter values. While thorough, it's computationally expensive and suffers from the curse of dimensionality as the number of hyperparameters grows."),
    ("What is random search?", "Random search samples hyperparameter combinations randomly from specified distributions. It's more efficient than grid search, especially when some hyperparameters don't significantly affect performance, and can discover better configurations with fewer evaluations."),
    ("What is Bayesian optimization?", "Bayesian optimization builds a probabilistic model of the objective function and uses it to select promising hyperparameters to evaluate next. It balances exploration and exploitation, efficiently finding good configurations with fewer expensive evaluations."),
    ("What is early stopping?", "Early stopping halts training when validation performance stops improving for a specified number of epochs (patience). It prevents overfitting by finding the point where the model generalizes best before it starts memorizing training data."),
    ("What is a learning rate schedule?", "Learning rate schedules adjust the learning rate during training, typically decreasing it over time. Common strategies include step decay, exponential decay, and cosine annealing. Proper scheduling improves convergence and final performance."),
    ("What is cosine annealing?", "Cosine annealing decreases learning rate following a cosine curve from initial to minimum value over a period. It provides smooth decay and can be restarted periodically (warm restarts) to escape local minima and improve optimization."),
    ("What is learning rate warmup?", "Learning rate warmup gradually increases the learning rate from zero to the target value over initial training steps. This prevents destabilization from large gradients early in training when the model is far from optimal, especially important for large batch sizes."),
    ("What is gradient clipping?", "Gradient clipping caps gradient magnitudes to a maximum value, preventing exploding gradients. It's essential for RNN training and stabilizes optimization when gradients occasionally spike. Common thresholds are clip by norm or clip by value."),
    ("What is gradient accumulation?", "Gradient accumulation computes gradients over multiple mini-batches before updating weights. This simulates larger batch sizes without increased memory, as gradients are summed across batches then averaged for the update. Useful for memory-constrained training."),
    ("What is mixed precision training?", "Mixed precision training uses both FP16 (fast, memory-efficient) and FP32 (stable, precise) arithmetic. Activations and gradients use FP16 while master weights stay in FP32. This accelerates training and reduces memory with minimal quality loss."),
    ("What is weight decay?", "Weight decay adds a penalty proportional to weight magnitudes to the loss function, encouraging smaller weights. It's equivalent to L2 regularization and prevents overfitting by discouraging complex models. Typical values are 0.0001 to 0.01."),
    ("What is label smoothing?", "Label smoothing replaces hard 0/1 targets with soft targets (e.g., 0.9/0.1), reducing model overconfidence. It acts as regularization, preventing overfitting and improving model calibration, especially important for classification with many classes."),
    ("What is curriculum learning?", "Curriculum learning trains models on progressively harder examples, starting with easy cases and gradually increasing difficulty. This mimics human learning, often improving convergence speed, final performance, and generalization compared to random ordering."),
]

# 7. EVALUATION & METRICS (20+ questions)
evaluation_metrics = [
    ("What is accuracy?", "Accuracy is the fraction of correct predictions: (TP + TN) / Total. While intuitive, it's misleading for imbalanced datasets where a naive baseline achieves high accuracy by always predicting the majority class."),
    ("What is precision?", "Precision is the fraction of true positives among all positive predictions: TP / (TP + FP). It answers 'Of all instances predicted positive, how many were actually positive?' High precision means few false positives."),
    ("What is recall?", "Recall (sensitivity) is the fraction of true positives among all actual positives: TP / (TP + FN). It answers 'Of all actual positive instances, how many did we find?' High recall means few false negatives."),
    ("What is the F1 score?", "The F1 score is the harmonic mean of precision and recall: 2 × (precision × recall) / (precision + recall). It balances both metrics, useful when you need a single score accounting for both false positives and false negatives."),
    ("What is the confusion matrix?", "A confusion matrix tabulates classification results showing true positives, false positives, true negatives, and false negatives. It provides detailed insight into model performance, revealing which classes are confused and error patterns."),
    ("What is ROC curve?", "The ROC (Receiver Operating Characteristic) curve plots true positive rate (recall) against false positive rate across all classification thresholds. It visualizes the tradeoff between sensitivity and specificity independent of class distribution."),
    ("What is AUC?", "AUC (Area Under the Curve) summarizes ROC curve performance in a single number from 0 to 1. AUC=0.5 means random guessing, AUC=1.0 means perfect classification. It measures discriminative ability across all possible thresholds."),
    ("What is perplexity?", "Perplexity measures how well a probability model predicts samples, commonly used for language models. It's the exponential of cross-entropy loss. Lower perplexity indicates better predictions, with perplexity of 1 meaning perfect prediction."),
    ("What is BLEU score?", "BLEU (Bilingual Evaluation Understudy) evaluates machine translation quality by comparing n-gram overlap between generated and reference translations. It ranges from 0 to 100, with higher scores indicating better translation quality."),
    ("What is mean average precision?", "Mean Average Precision (mAP) evaluates object detection and information retrieval by averaging precision at different recall levels. It accounts for both detection accuracy and localization quality, providing a comprehensive performance metric."),
]

# Compile all questions
all_qa_pairs = (
    ml_fundamentals +
    neural_networks +
    computer_vision +
    nlp +
    lora_finetuning +
    training_optimization +
    evaluation_metrics
)

def create_variations(question, answer):
    """Create multiple format variations of each Q&A pair"""
    variations = [
        f"Q: {question}\nA: {answer}",
        f"Question: {question}\nAnswer: {answer}",
        f"User: {question}\nAssistant: {answer}",
        f"{question}\n\n{answer}",
    ]
    return variations

def generate_dataset():
    """Generate comprehensive training dataset with variations"""
    print("="*80)
    print(" Generating High-Quality ML/AI Training Dataset")
    print("="*80)
    print(f"\nBase Q&A pairs: {len(all_qa_pairs)}")

    # Generate variations
    all_examples = []
    for question, answer in all_qa_pairs:
        variations = create_variations(question, answer)
        for var in variations:
            all_examples.append({"text": var})

    print(f"Total examples (with variations): {len(all_examples)}")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_examples)

    # Split: 80% train, 10% valid, 10% test
    total = len(all_examples)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_set = all_examples[:train_end]
    valid_set = all_examples[train_end:valid_end]
    test_set = all_examples[valid_end:]

    # Write datasets
    print(f"\nWriting datasets...")
    with open("data/train.jsonl", "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    with open("data/valid.jsonl", "w") as f:
        for item in valid_set:
            f.write(json.dumps(item) + "\n")

    with open("data/test.jsonl", "w") as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")

    print(f"\n✓ Training examples:   {len(train_set):4d}")
    print(f"✓ Validation examples: {len(valid_set):4d}")
    print(f"✓ Test examples:       {len(test_set):4d}")
    print(f"✓ Total:               {total:4d}")

    # Calculate dataset statistics
    avg_length = sum(len(ex["text"]) for ex in all_examples) / len(all_examples)
    print(f"\nAverage text length: {avg_length:.0f} characters")
    print(f"\nDataset topics:")
    print(f"  - ML Fundamentals:       {len(ml_fundamentals)} questions")
    print(f"  - Neural Networks:       {len(neural_networks)} questions")
    print(f"  - Computer Vision:       {len(computer_vision)} questions")
    print(f"  - NLP:                   {len(nlp)} questions")
    print(f"  - LoRA & Fine-tuning:    {len(lora_finetuning)} questions")
    print(f"  - Training & Optimization: {len(training_optimization)} questions")
    print(f"  - Evaluation Metrics:    {len(evaluation_metrics)} questions")

    print("\n" + "="*80)
    print(" Dataset generation complete!")
    print("="*80)
    print("\nReady for training!")

if __name__ == "__main__":
    generate_dataset()
