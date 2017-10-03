class NeuralNetwork;

# input/output layer config
has List @training_data;
has Int num_examples; #TODO init this as @training_data.elems
has Int nn_input_dim = 2;
has Int nn_output_dim = 2;

# gradient descent params
has Num epsilon = 0.01;
has Num reg_lambda = 0.01;

sub calculate_loss($model) {
  my ($W1, $b1, $W2, $b2) = $model<W1 b1 W2 b2>;
  my $z1 = @training_data
  #  z1 = X.dot(W1) + b1
  # a1 = np.tanh(z1)
  # z2 = a1.dot(W2) + b2
  # exp_scores = np.exp(z2)
  # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  # # Calculating the loss
  # corect_logprobs = -np.log(probs[range(num_examples), y])
  # data_loss = np.sum(corect_logprobs)
  # # Add regulatization term to loss (optional)
  # data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
  # return 1./num_examples * data_loss
}

sub predict($model, $x) {
  # # Helper function to predict an output (0 or 1)
  # def predict(model, x):
  my ($W1, $b1, $W2, $b2) = model<W1 b1 W2 b2>;
  #     # Forward propagation
  my $z1 = $x.dot($W1) + $b1;
  #     a1 = np.tanh(z1)
  #     z2 = a1.dot(W2) + b2
  #     exp_scores = np.exp(z2)
  #     probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  #     return np.argmax(probs, axis=1)
}

sub build_model() {
  #   # This function learns parameters for the neural network and returns the model.
  # # - nn_hdim: Number of nodes in the hidden layer
  # # - num_passes: Number of passes through the training data for gradient descent
  # # - print_loss: If True, print the loss every 1000 iterations
  # def build_model(nn_hdim, num_passes=20000, print_loss=False):
      
  #     # Initialize the parameters to random values. We need to learn these.
  #     np.random.seed(0)
  #     W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
  #     b1 = np.zeros((1, nn_hdim))
  #     W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
  #     b2 = np.zeros((1, nn_output_dim))
  
  #     # This is what we return at the end
  #     model = {}
      
  #     # Gradient descent. For each batch...
  #     for i in xrange(0, num_passes):
  
  #         # Forward propagation
  #         z1 = X.dot(W1) + b1
  #         a1 = np.tanh(z1)
  #         z2 = a1.dot(W2) + b2
  #         exp_scores = np.exp(z2)
  #         probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  
  #         # Backpropagation
  #         delta3 = probs
  #         delta3[range(num_examples), y] -= 1
  #         dW2 = (a1.T).dot(delta3)
  #         db2 = np.sum(delta3, axis=0, keepdims=True)
  #         delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
  #         dW1 = np.dot(X.T, delta2)
  #         db1 = np.sum(delta2, axis=0)
  
  #         # Add regularization terms (b1 and b2 don't have regularization terms)
  #         dW2 += reg_lambda * W2
  #         dW1 += reg_lambda * W1
  
  #         # Gradient descent parameter update
  #         W1 += -epsilon * dW1
  #         b1 += -epsilon * db1
  #         W2 += -epsilon * dW2
  #         b2 += -epsilon * db2
          
  #         # Assign new parameters to the model
  #         model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
          
  #         # Optionally print the loss.
  #         # This is expensive because it uses the whole dataset, so we don't want to do it too often.
  #         if print_loss and i % 1000 == 0:
  #           print "Loss after iteration %i: %f" %(i, calculate_loss(model))
      
  #     return model
}