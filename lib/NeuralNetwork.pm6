class NeuralNetwork {

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
}