
import numpy as np
import time


""" Loss and gradient """

def loss(y, y_bar, W, W_bar):
  term_1 = np.sum(W * (y - y_bar[:, None])**2)
  term_2 = 0.5 * np.sum(W_bar * (y[:, None] - y)**2)
  return term_1 + term_2


def gradient(y, y_bar, W, W_bar):
  grad_1 = 2 * np.sum(W * (y - y_bar[:, None]), axis=0)
  grad_2 = 2 * np.sum(W_bar * (y[:, None] - y), axis=1)
  return grad_1 + grad_2


def gradient_block(b, y, y_bar, W, W_bar):
  if len(b) == 1:
    b0 = b[0]
    grad_1 = 2 * np.sum(W[:, b0] * (y[b0] - y_bar))
    grad_2 = 2 * np.sum(W_bar[b0, :] * (y - y[b0]))
  elif len(b) > 1:
    grad_1 = 2 * np.sum(W[:, b] * (y[b] - y_bar[:, None]), axis=0)
    grad_2 = 2 * np.sum(W_bar[b, :] * (y[None, :] - y[b][:, None]), axis=1)
  return grad_1 + grad_2


""" Accuracy """

def calculate_accuracy(y, y_true):
  y_pred = np.sign(y)
  correct_preds = np.sum(y_pred == y_true)
  return correct_preds / len(y_true)


""" Early stopping """

def early_stopping_gradient(grad_norm, epsilon, k, loss_value, accuracy):
  if grad_norm < epsilon:
    return {
      "stop": True,
      "message": (f"Stopped early: gradient norm < {epsilon}.\n"
                  f"Converged in {k} iterations, ||grad|| = {grad_norm:.2e}, "
                  f"loss = {loss_value:.3f}, accuracy = {accuracy*100:.3f}%")
    }
  return {"stop": False}


def early_stopping_max_accuracy(accuracy, accuracy_threshold, max_accuracy, best_iteration, loss_history, grad_history):
  if accuracy * 100 >= accuracy_threshold:
    return {
      "stop": True,
      "message": (f"Stopped early: accuracy >= {accuracy_threshold}%\n"
                  f"Best accuracy: {max_accuracy*100:.3f}% at iteration {best_iteration+1}.\n"
                  f"Iter {best_iteration+1}: loss = {loss_history[best_iteration]:.3f}, "
                  f"||grad|| = {grad_history[best_iteration]:.3f}")
    }
  return {"stop": False}


def early_stopping_stagnant_accuracy(accuracy, last_n_accuracies, stagnant_accuracy_max_iter,
                                      max_accuracy, best_iteration, loss_history, grad_history):
  last_n_accuracies.append(accuracy)
  if len(last_n_accuracies) > stagnant_accuracy_max_iter:
    last_n_accuracies.pop(0)
  if len(last_n_accuracies) == stagnant_accuracy_max_iter:
    if all(last_n_accuracies[i] >= last_n_accuracies[i+1] for i in range(stagnant_accuracy_max_iter-1)):
      return {
        "stop": True,
        "message": (f"Stopped early: accuracy decreased or stayed the same for {stagnant_accuracy_max_iter} iterations.\n"
                    f"Best accuracy: {max_accuracy*100:.3f}% at iteration {best_iteration+1}.\n"
                    f"Iter {best_iteration+1}: loss = {loss_history[best_iteration]:.3f}, "
                    f"||grad|| = {grad_history[best_iteration]:.3f}")
      }
  return {"stop": False}


""" Algorithms """

def gradient_descent(y_bar, W, W_bar, y_true,
                     alpha=1.0, max_iter=1000,
                     epsilon=1e-6,
                     stagnant_accuracy_stop=True, stagnant_accuracy_max_iter=20,
                     accuracy_threshold=98.0,
                     verbose=True, seed=42):
  # Initialize logs
  loss_history = []
  grad_history = []
  accuracy_history = []
  time_history = []
  stopped_early_acc = False
  stopped_early_grad = False

  # Initialize accuracy tracker
  if stagnant_accuracy_stop:
    last_n_accuracies = []
  max_accuracy = -float('inf')
  best_iteration = -1

  # Initialize labels
  np.random.seed(seed)
  y = np.random.uniform(-1.0, 1.0, size=len(y_true))

  start_time = time.time()

  for k in range(max_iter):
    iter_start = time.time()

    """ Gradient descent step """
    grad = gradient(y, y_bar, W, W_bar)
    direction = -grad
    y += alpha * direction

    """ Trackers """
    # Gradient and loss
    grad_norm = np.linalg.norm(grad)
    loss_value = loss(y, y_bar, W, W_bar)

    # Accuracy
    accuracy = calculate_accuracy(y, y_true)
    if accuracy > max_accuracy:
      max_accuracy = accuracy
      best_iteration = k

    loss_history.append(loss_value)
    accuracy_history.append(accuracy)
    grad_history.append(grad_norm)

    """ Early stopping """
    # Stagnant accuracy
    if stagnant_accuracy_stop:
      result = early_stopping_stagnant_accuracy(accuracy, last_n_accuracies, stagnant_accuracy_max_iter, max_accuracy, best_iteration, loss_history, grad_history)
      if result["stop"]:
        print(result["message"])
        stopped_early_acc = True
        break

    # Accuracy threshold
    result = early_stopping_max_accuracy(accuracy, accuracy_threshold, max_accuracy, best_iteration, loss_history, grad_history)
    if result["stop"]:
      print(result["message"])
      stopped_early_acc = True
      break

    # Gradient norm
    result = early_stopping_gradient(grad_norm, epsilon, k, loss_value, accuracy)
    if result["stop"]:
      print(result["message"])
      stopped_early_grad = True
      break

    """ Verbose iteration logs """
    time_history.append(time.time() - iter_start)
    if verbose and k % 10 == 0:
      print(f"Iter {k:3d}: loss = {loss_value:5.4f}, ||grad|| = {grad_norm:5.4f}, alpha = {alpha:5.6f}, accuracy = {accuracy*100:5.3f}%")

  """ Final logs """
  if not stopped_early_grad and not stopped_early_acc:
    print(f"Maximum number of iterations reached.")
    print(f"Best accuracy: {max_accuracy*100:.3f}% at iteration {best_iteration+1}.")
    print(f"Iter {best_iteration+1}: loss = {loss_history[best_iteration]:.3f}, ||grad|| = {grad_history[best_iteration]:.3f}")

  total_time = time.time() - start_time
  time_history.append(total_time)
  print(f"Total CPU time: {total_time:.4f} seconds.")

  return y, loss_history, grad_history, accuracy_history, time_history


def bcgd_gs(y_bar, W, W_bar, y_true,
            alpha=1.0, max_iter=1000,
            epsilon=1e-6, block_size=1,
            cache_gradient=False,
            stagnant_accuracy_stop=True, stagnant_accuracy_max_iter=20,
            accuracy_threshold=97.0,
            verbose=True, seed=42):
  # Initialize logs
  loss_history = []
  grad_history = []
  if cache_gradient:
    cached_grad = None
  accuracy_history = []
  time_history = []
  stopped_early_acc = False
  stopped_early_grad = False

  # Initialize accuracy tracker
  if stagnant_accuracy_stop:
    last_n_accuracies = []
  max_accuracy = -float('inf')
  best_iteration = -1

  # Initialize labels
  np.random.seed(seed)
  y = np.random.uniform(-1.0, 1.0, size=len(y_true))

  start_time = time.time()

  for k in range(max_iter):
    iter_start = time.time()

    """ BCGD step with Gauss-Southwell rule """
    if cache_gradient:
      if k == 0:
        grad = gradient(y, y_bar, W, W_bar) # Compute full gradient on first iteration
        cached_grad = grad.copy() # Cache for subsequent iterations
      else:
        grad = cached_grad.copy()

      if block_size == 1:
        b = [np.argmax(np.abs(grad))]
      elif block_size > 1:
        b = np.argsort(np.abs(grad))[-block_size:]

      grad[b] = gradient_block(b, y, y_bar, W, W_bar) # Calculate gradient only for the coordinates in the block
      y[b] -= alpha * grad[b] # Update the coordinate(s)
      cached_grad[b] = grad[b] # Cache gradient for coordinate(s) in 'b'

    else:
      grad = gradient(y, y_bar, W, W_bar)
      if block_size == 1:
        b = [np.argmax(np.abs(grad))]
      elif block_size > 1:
        b = np.argsort(np.abs(grad))[-block_size:]
      y[b] -= alpha * grad[b]

    """ Trackers """
    # Gradient and loss
    grad_norm = np.linalg.norm(grad)
    loss_value = loss(y, y_bar, W, W_bar)

    # Accuracy
    accuracy = calculate_accuracy(y, y_true)
    if accuracy > max_accuracy:
      max_accuracy = accuracy
      best_iteration = k

    loss_history.append(loss_value)
    accuracy_history.append(accuracy)
    grad_history.append(grad_norm)

    """ Early stopping """
    # Stagnant accuracy
    if stagnant_accuracy_stop:
      result = early_stopping_stagnant_accuracy(accuracy, last_n_accuracies, stagnant_accuracy_max_iter, max_accuracy, best_iteration, loss_history, grad_history)
      if result["stop"]:
        print(result["message"])
        stopped_early_acc = True
        break

    # Accuracy threshold
    result = early_stopping_max_accuracy(accuracy, accuracy_threshold, max_accuracy, best_iteration, loss_history, grad_history)
    if result["stop"]:
      print(result["message"])
      stopped_early_acc = True
      break

    # Gradient norm
    result = early_stopping_gradient(grad_norm, epsilon, k, loss_value, accuracy)
    if result["stop"]:
      print(result["message"])
      stopped_early_grad = True
      break

    """ Verbose iteration logs """
    time_history.append(time.time() - iter_start)
    if verbose and k % 10 == 0:
      print(f"Iter {k:3d}: loss = {loss_value:5.4f}, ||grad|| = {grad_norm:5.4f}, alpha = {alpha:5.6f}, accuracy = {accuracy*100:5.3f}%")

  """ Final logs """
  if not stopped_early_grad and not stopped_early_acc:
    print(f"Maximum number of iterations reached.")
    print(f"Best accuracy: {max_accuracy*100:.3f}% at iteration {best_iteration+1}.")
    print(f"Iter {best_iteration+1}: loss = {loss_history[best_iteration]:.3f}, ||grad|| = {grad_history[best_iteration]:.3f}")

  total_time = time.time() - start_time
  time_history.append(total_time)
  print(f"Total CPU time: {total_time:.4f} seconds.")

  return y, loss_history, grad_history, accuracy_history, time_history
