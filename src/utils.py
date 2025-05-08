
# If recreating animated 3D plots, run !pip install kaleido first. Might need to restart the session if working in Colab.
import os
import numpy as np
import pandas as pd
import json
import math
import imageio
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
import shutil
import kaleido


""" Data preprocessing """

def z_scores(data):
  mean = np.mean(data, axis=0)
  std = np.std(data, axis=0)
  return (data - mean) / std


def generate_data(n, n_features, centers, std, proportion_labeled, normalize=True, seed=42):
  np.random.seed(seed)

  n_labeled = math.floor(n * proportion_labeled)

  X, y = make_blobs(n_samples=n, n_features=n_features, centers=centers, cluster_std=std, random_state=seed)
  y = np.where(y == 0, -1, 1)

  idx = np.random.permutation(n)
  X, y = X[idx], y[idx]

  l_idx = np.arange(n_labeled)
  u_idx = np.arange(n_labeled, n)

  if normalize:
    X_std = z_scores(X)
    X_l, X_u = X_std[l_idx], X_std[u_idx]
    y_l, y_u = y[l_idx], y[u_idx]
    return X_l, y_l, X_u, y_u

  X_l, X_u = X[l_idx], X[u_idx]
  y_l, y_u = y[l_idx], y[u_idx]
  return X_l, y_l, X_u, y_u


def import_data(data_path, features, labels, preprocess=True):
  df = pd.read_csv(f"{data_path}/riceClassification.csv")
  data = df[features + labels].copy()

  if preprocess:
    label_col = labels[0]
    data[label_col] = data[label_col].astype(int) * 2 - 1
    class_pos = data[data[label_col] == 1]
    class_neg = data[data[label_col] == -1]

    class_pos_sample = class_pos.sample(n=5000, random_state=42)
    class_neg_sample = class_neg.sample(n=5000, random_state=42)

    balanced_data = pd.concat([class_pos_sample, class_neg_sample])
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    x = balanced_data[features]
    y = balanced_data[[label_col]]
    standardized_features = z_scores(x)

    processed_data = pd.concat([standardized_features, y], axis=1)
    return processed_data

  return data


def labeled_unlabeled_split(data, features, labels, n_labeled=1000, seed=42):
  np.random.seed(seed)
  label_col = labels[0]
  all_idx = np.arange(len(data))
  l_idx = np.random.choice(all_idx, size=n_labeled, replace=False)
  u_idx = np.setdiff1d(all_idx, l_idx)

  l_data = data.iloc[l_idx]
  u_data = data.iloc[u_idx]

  X_l = l_data[features].values
  y_l = l_data[label_col].values

  X_u = u_data[features].values
  y_u = u_data[label_col].values

  return X_l, y_l, X_u, y_u


""" Store algorithm results """

def structure_results(**named_results):
  keys = ['y_pred', 'loss', 'grad', 'accuracy', 'time']
  results = {name: dict(zip(keys, res)) for name, res in named_results.items()}
  return results


def convert_ndarray_to_list(data):
  if isinstance(data, np.ndarray):
    return data.tolist()
  return data


def convert_list_to_ndarray(data):
  if isinstance(data, list):
    return np.array(data)
  return data


""" Plots """

def plot_3d_data(X_l, y_l, X_u, title, x_lab, y_lab, z_lab):
  palette=dict(
    orange='#faa200',
    vermillion='#f3640d',
    sky_blue='#00b7ec',
    blue='#0077b8',
  )

  unlabeled = go.Scatter3d(
    x=X_u[:, 0], y=X_u[:, 1], z=X_u[:, 2],
    mode='markers',
    marker=dict(
      size=5,
      color='gray',
      opacity=0.2,
      symbol='circle',
      line=dict(color='black', width=1.0)
    ),
    name='Unlabeled'
  )

  class_pos = go.Scatter3d(
    x=X_l[y_l == 1, 0], y=X_l[y_l == 1, 1], z=X_l[y_l == 1, 2],
    mode='markers',
    marker=dict(
      size=5,
      color=palette['orange'],
      opacity=1.0,
      symbol='circle',
      line=dict(color=palette['vermillion'], width=1.0)
    ),
    name='Class +1'
  )

  class_neg = go.Scatter3d(
    x=X_l[y_l == -1, 0], y=X_l[y_l == -1, 1], z=X_l[y_l == -1, 2],
    mode='markers',
    marker=dict(
      size=5,
      color=palette['sky_blue'],
      opacity=1.0,
      symbol='circle',
      line=dict(color=palette['blue'], width=1.0)
    ),
    name='Class  -1'
  )

  # Create 3D Layout
  layout = go.Layout(
    title=title,
    scene=dict(
      xaxis_title=x_lab,
      yaxis_title=y_lab,
      zaxis_title=z_lab,
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(
      x=0.8,
      y=0.9,
      traceorder='normal',
      font=dict(size=14),
      borderwidth=1
    ),
  )

  fig = go.Figure(data=[unlabeled, class_pos, class_neg], layout=layout)
  fig.show()


def plot_decision_boundary_3d(X_l, X_u, y_l, y_u, y_pred, title="Decision Boundary", grid_size=25, frames=60):
  # Define the color palette
  palette = dict(
    orange='#faa200',
    vermillion='#f3640d',
    sky_blue='#00b7ec',
    blue='#0077b8',
  )

  # Combine data for bounds
  X_all = np.vstack([X_l, X_u])

  # Get bounds
  x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
  y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
  z_min, z_max = X_all[:, 2].min() - 1, X_all[:, 2].max() + 1

  # Create a 3D grid
  x_vals = np.linspace(x_min, x_max, grid_size)
  y_vals = np.linspace(y_min, y_max, grid_size)
  z_vals = np.linspace(z_min, z_max, grid_size)
  Xg, Yg, Zg = np.meshgrid(x_vals, y_vals, z_vals)

  # Flatten grid
  grid_points = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]

  # Fit a KNN model on labeled data for prediction
  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(X_l, y_l)
  predictions = knn.predict(grid_points)
  predictions = predictions.reshape(Xg.shape)

  # Create isosurface plot for decision boundary
  fig = go.Figure(data=go.Isosurface(
    x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
    value=predictions.ravel(),
    isomin=0, isomax=0, surface_count=1,
    colorscale='RdBu', opacity=0.9,
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=False
  ))

  # Add labeled data points with specific color and markers (matching palette)
  class_pos = go.Scatter3d(
    x=X_l[y_l == 1, 0], y=X_l[y_l == 1, 1], z=X_l[y_l == 1, 2],
    mode='markers',
    marker=dict(
      size=5,
      color=palette['orange'],
      opacity=1.0,
      symbol='circle',
      line=dict(color=palette['vermillion'], width=1.0)
    ),
    name='Class +1'
  )

  class_neg = go.Scatter3d(
    x=X_l[y_l == -1, 0], y=X_l[y_l == -1, 1], z=X_l[y_l == -1, 2],
    mode='markers',
    marker=dict(
      size=5,
      color=palette['sky_blue'],
      opacity=1.0,
      symbol='circle',
      line=dict(color=palette['blue'], width=1.0)
    ),
    name='Class -1'
  )

  y_u_pred = np.sign(y_pred)
  colors = np.array([palette['orange'] if label == 1 else palette['sky_blue'] for label in y_u_pred])

  # Set outlines based on the predicted class
  outlines = np.array([palette['vermillion'] if label == 1 else palette['blue'] for label in y_u_pred])
  unlabeled_pred = go.Scatter3d(
    x=X_u[:, 0], y=X_u[:, 1], z=X_u[:, 2],
    mode='markers',
    marker=dict(
      size=5,
      color=colors,
      opacity=1.0,
      symbol='circle-open',
      line=dict(color=outlines, width=2.0)
    ),
    name='Predicted Unlabeled'
  )

  # Combine all data for the figure
  fig.add_trace(class_pos)
  fig.add_trace(class_neg)
  fig.add_trace(unlabeled_pred)

  # Update layout with axis titles
  fig.update_layout(
    title=title,
    scene=dict(
      xaxis_title="X",
      yaxis_title="Y",
      zaxis_title="Z"
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(
      x=0.8,
      y=0.9,
      traceorder='normal',
      font=dict(size=14),
      borderwidth=1
    ),
  )

  fig.show()


def plot_decision_boundary_3d_animated(X_l, X_u, y_l, y_u, y_pred, title="Decision Boundary", grid_size=25, frames=36):
  def _render_frame(args):
    i, base_fig, angle_rad, frames_dir = args
    fig = base_fig
    camera = dict(
      eye=dict(
        x=2 * np.cos(angle_rad),
        y=2 * np.sin(angle_rad),
        z=1.25
      )
    )
    fig.update_layout(scene_camera=camera)
    filepath = os.path.join(frames_dir, f"frame_{i:03d}.png")
    pio.write_image(fig, filepath, format='png', scale=1)  # Scale=1 is faster
    return filepath

  palette = dict(
    orange='#faa200',
    vermillion='#f3640d',
    sky_blue='#00b7ec',
    blue='#0077b8',
  )

  X_all = np.vstack([X_l, X_u])
  x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
  y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1
  z_min, z_max = X_all[:, 2].min() - 1, X_all[:, 2].max() + 1

  x_vals = np.linspace(x_min, x_max, grid_size)
  y_vals = np.linspace(y_min, y_max, grid_size)
  z_vals = np.linspace(z_min, z_max, grid_size)
  Xg, Yg, Zg = np.meshgrid(x_vals, y_vals, z_vals)
  grid_points = np.c_[Xg.ravel(), Yg.ravel(), Zg.ravel()]

  knn = KNeighborsClassifier(n_neighbors=1)
  knn.fit(X_l, y_l)
  predictions = knn.predict(grid_points).reshape(Xg.shape)

  isosurface = go.Isosurface(
    x=Xg.ravel(), y=Yg.ravel(), z=Zg.ravel(),
    value=predictions.ravel(),
    isomin=0, isomax=0, surface_count=1,
    colorscale='RdBu', opacity=0.9,
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=False
  )

  class_pos = go.Scatter3d(
    x=X_l[y_l == 1, 0], y=X_l[y_l == 1, 1], z=X_l[y_l == 1, 2],
    mode='markers',
    marker=dict(size=5, color=palette['orange'],
                opacity=1.0, line=dict(color=palette['vermillion'], width=1.0)),
    name='Class +1'
  )

  class_neg = go.Scatter3d(
    x=X_l[y_l == -1, 0], y=X_l[y_l == -1, 1], z=X_l[y_l == -1, 2],
    mode='markers',
    marker=dict(size=5, color=palette['sky_blue'],
                opacity=1.0, line=dict(color=palette['blue'], width=1.0)),
    name='Class -1'
  )

  y_u_pred = np.sign(y_pred)
  colors = np.array([palette['orange'] if label == 1 else palette['sky_blue'] for label in y_u_pred])
  outlines = np.array([palette['vermillion'] if label == 1 else palette['blue'] for label in y_u_pred])

  unlabeled_pred = go.Scatter3d(
    x=X_u[:, 0], y=X_u[:, 1], z=X_u[:, 2],
    mode='markers',
    marker=dict(
      size=5,
      color=colors,
      opacity=1.0,
      symbol='circle-open',
      line=dict(color=outlines, width=2.0)
    ),
    name='Predicted Unlabeled'
  )

  fig = go.Figure(data=[isosurface, class_pos, class_neg, unlabeled_pred])
  fig.update_layout(
    title=title,
    scene=dict(
      xaxis_title="X",
      yaxis_title="Y",
      zaxis_title="Z",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    legend=dict(
      x=0.8,
      y=0.9,
      traceorder='normal',
      font=dict(size=14),
      borderwidth=1
    ),
  )

  # Create frame directory
  frames_dir = "frames"
  os.makedirs(frames_dir, exist_ok=True)

  # Precompute rotation angles
  angles = [np.radians(85) * np.sin(2 * np.pi * i / frames) for i in range(frames)]

  # Parallel rendering
  args = [(i, fig.to_dict(), angle, frames_dir) for i, angle in enumerate(angles)]

  def fig_from_dict(d): return go.Figure(d)  # Workaround for Plotly immutability

  with Pool(min(cpu_count(), 4)) as pool:  # Limit to 4 processes to avoid Colab crash
    _render_args = [(i, fig_from_dict(fig_dict), angle, frames_dir) for i, fig_dict, angle, _ in args]
    pool.map(_render_frame, _render_args)

  # Create GIF
  with imageio.get_writer("decision_boundary_looped.gif", mode="I", duration=0.1, loop=0) as writer:
    for i in range(frames):
      image = imageio.imread(os.path.join(frames_dir, f"frame_{i:03d}.png"))
      writer.append_data(image)

  shutil.rmtree(frames_dir)
  print("GIF saved as 'decision_boundary_looped.gif'")


def plot_algorithm_comparison(results, toy=False, max_iter=1000):
  def pad_to_max_iter(history, max_iter):
    history = history.tolist() if isinstance(history, np.ndarray) else history
    return history + [np.nan] * (max_iter - len(history)) if len(history) < max_iter else history[:max_iter]

  # Configuration for each algorithm
  if toy:
    algorithms = {
      'gd_res_toy': {'label': 'GD', 'color': '#f3640d'},
      'bcgd_1_d_non_cached_res_toy': {'label': 'BCGD-GS 1-D (non-cached)', 'color': '#10a53d'},
      'bcgd_1_d_cached_res_toy': {'label': 'BCGD-GS 1-D (cached)', 'color': '#3a5e8c'},
    }

  else:
    algorithms = {
      'gd_res': {'label': 'GD', 'color': '#f3640d'},
      'bcgd_1_d_non_cached_res': {'label': 'BCGD-GS 1-D (non-cached)', 'color': '#10a53d'},
      'bcgd_1_d_cached_res': {'label': 'BCGD-GS 1-D (cached)', 'color': '#2f9aa0'},
      'bcgd_10_d_cached_res': {'label': 'BCGD-GS 10-D (cached)', 'color': '#3a5e8c'},
      'bcgd_100_d_cached_res': {'label': 'BCGD-GS 100-D (cached)', 'color': '#541352'}
    }

  accuracy_data = {}
  loss_data = {}
  time_data = {}
  total_times = {}
  total_iters = {}

  for key, config in algorithms.items():
    data = results[key]
    accuracy_data[key] = pad_to_max_iter(data['accuracy'], max_iter)
    loss_data[key] = pad_to_max_iter(data['loss'], max_iter)
    time_data[key] = pad_to_max_iter(data['time'][:-1], max_iter)
    total_times[key] = data['time'][-1]
    total_iters[key] = len(data['accuracy']) 

  # Set up a 2x2 grid of subplots
  fig, axs = plt.subplots(2, 2, figsize=(14, 10))

  # Accuracy vs Iteration
  for key, config in algorithms.items():
    axs[0, 0].plot(range(max_iter), accuracy_data[key], label=f"{config['label']}", color=config['color'], lw=2.5)
  axs[0, 0].set_xlabel('Iterations')
  axs[0, 0].set_ylabel('Accuracy')
  axs[0, 0].set_title('Accuracy vs. Iterations')
  axs[0, 0].legend(loc='center right', frameon=True)

  # Loss vs Iteration
  for key, config in algorithms.items():
    axs[0, 1].plot(range(max_iter), loss_data[key], label=f"{config['label']}", color=config['color'], linestyle='dashed', lw=2.5)
  axs[0, 1].set_xlabel('Iterations')
  axs[0, 1].set_ylabel('Loss')
  axs[0, 1].set_title('Loss vs. Iterations')
  axs[0, 1].legend(loc='center right', frameon=True)

  # CPU Time per Iteration
  for key, config in algorithms.items():
    axs[1, 0].plot(range(max_iter), time_data[key], label=f"{config['label']}", color=config['color'])
  axs[1, 0].set_xlabel('Iterations')
  axs[1, 0].set_ylabel('CPU Time (seconds)')
  axs[1, 0].set_title('CPU Time vs. Iterations')
  axs[1, 0].legend(loc='upper right', frameon=True)

  # Total CPU Time and Total Iterations per Algorithm
  names = [config['label'] for config in algorithms.values()]
  times = [total_times[key] for key in algorithms]
  iterations = [total_iters[key] for key in algorithms]
  colors = [config['color'] for config in algorithms.values()]

  # Plotting bars with different fill patterns
  bar_width = 0.35
  space_between_bars = 0.05
  bars1 = axs[1, 1].bar([i - bar_width / 2 - space_between_bars / 2 for i in range(len(names))], 
                        times, bar_width, color=colors, label='CPU Time')
  bars2 = axs[1, 1].bar([i + bar_width / 2 + space_between_bars / 2 for i in range(len(names))], 
                        iterations, bar_width, color=colors, hatch='//', edgecolor='white', label='Iterations')

  axs[1, 1].set_ylabel('Value')
  axs[1, 1].set_title('Total CPU Time and Total Iterations per Algorithm')
  axs[1, 1].set_xticks(range(len(names)))
  axs[1, 1].set_xticklabels(names, rotation=45, ha='right')

  # Annotate bars with values
  for bar, time in zip(bars1, times):
    axs[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
  for bar, iteration in zip(bars2, iterations):
    axs[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{iteration}', ha='center', va='bottom', fontsize=10)

  axs[1, 1].legend(loc='upper left', frameon=True)

  plt.tight_layout()
  plt.show()


""" Similarity and Lipschitz constants """

def similarity_matrix(X1, X2, epsilon=1e-6, normalize_rows=True, ensure_symmetry=False):
  matrix = 1/(cdist(X1, X2, 'euclidean') + epsilon)
  if normalize_rows:
    matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
  if ensure_symmetry:
    matrix = (matrix + matrix.T) / 2
  return matrix


def lipschitz(W, W_bar):
  # Hessian First term
  W_sum = np.sum(W, axis=0)
  D = diags(W_sum) # Shape (unlabeled, unlabeled)
  term_1 = 2 * D

  # Hessian Second term
  W_bar = csr_matrix(W_bar)
  W_bar_sum = np.array(W_bar.sum(axis=1)).flatten() # Shape (unlabeled, ), ready for diagonal
  D_bar = diags(W_bar_sum)
  L_bar = D_bar - W_bar # Shape (unlabeled, unlabeled)
  term_2 = 2 * L_bar

  # Full Hessian
  H = term_1 + term_2

  # Max lambda (Lispchitz contant)
  L = eigsh(H, k=1, which='LM', return_eigenvectors=False)[0]
  alpha = 1.0 / L

  return L, alpha
