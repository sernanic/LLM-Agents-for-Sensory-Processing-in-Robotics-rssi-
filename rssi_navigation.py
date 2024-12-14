"""
RSSI Navigation System with Uncertainty Estimation
===============================================

This module implements an RSSI-based navigation system using advanced machine learning techniques:
1. Gaussian Process Regression for RSSI prediction
2. SIFT algorithm for uncertainty estimation
3. Action selection based on predicted RSSI values and uncertainty

The system maintains a sequence of RSSI measurements and their corresponding positions,
using this history to make predictions about future RSSI values and optimal movements.

Key Components:
-------------
- RSSI Prediction: Uses Gaussian Process Regression to predict future signal strengths
- Uncertainty Estimation: Implements SIFT algorithm for reliable uncertainty quantification
- Action Selection: Combines predictions and uncertainty for optimal navigation decisions
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from typing import List, Tuple, Dict
import torch
import torch.nn as nn

class RSSINavigator:
    """
    Main class for RSSI-based navigation with uncertainty estimation.
    
    This class combines several machine learning techniques to achieve robust
    navigation in environments with varying RSSI signal strengths:
    
    1. Maintains a history of RSSI measurements and positions
    2. Predicts future RSSI values using Gaussian Process Regression
    3. Estimates uncertainty using the SIFT algorithm
    4. Selects optimal actions based on predictions and uncertainty
    
    Attributes:
        grid_size (int): Size of the navigation grid (default: 11x11)
        sequence_length (int): Number of historical measurements to maintain
        actions (List[str]): Available navigation actions
        uncertainty_threshold (float): Threshold for acceptable uncertainty
    """
    
    def __init__(self, grid_size: int = 11, sequence_length: int = 10):
        """
        Initialize the RSSI Navigator with specified parameters.
        
        Args:
            grid_size: Size of the navigation grid (NxN)
            sequence_length: Number of historical measurements to maintain
        """
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        self.actions = ['left', 'right', 'straight', 'back']
        
        # Initialize Gaussian Process for RSSI prediction
        # RBF kernel captures smooth variations in RSSI values
        # WhiteKernel accounts for measurement noise
        self.kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)
        
        # Threshold for determining when uncertainty is too high
        self.uncertainty_threshold = 0.5
        
        # Storage for historical measurements
        self.rssi_sequence: List[float] = []
        self.position_sequence: List[Tuple[int, int]] = []
        
        # Track visited positions and their visit counts
        self.visited_positions = {}
        
        # Exploration parameters
        self.exploration_weight = 0.3  # Weight for exploration bonus
        self.visit_decay = 0.95  # Decay factor for repeated visits
        
    def update_sequence(self, rssi_value: float, position: Tuple[int, int]):
        """
        Update the historical sequences with new measurements.
        
        This method maintains a sliding window of recent measurements,
        keeping only the most recent sequence_length values.
        
        Args:
            rssi_value: New RSSI measurement
            position: Current position (x, y) where measurement was taken
        """
        self.rssi_sequence.append(rssi_value)
        self.position_sequence.append(position)
        
        # Maintain fixed sequence length using sliding window
        if len(self.rssi_sequence) > self.sequence_length:
            self.rssi_sequence = self.rssi_sequence[-self.sequence_length:]
            self.position_sequence = self.position_sequence[-self.sequence_length:]
    
    def predict_next_rssi(self, current_position: Tuple[int, int]) -> Tuple[float, float]:
        """
        Predict the next RSSI value and its uncertainty using Gaussian Process Regression.
        
        The prediction process:
        1. Uses historical position-RSSI pairs for training
        2. Fits a Gaussian Process model
        3. Predicts RSSI value and uncertainty for the current position
        
        Args:
            current_position: Position (x, y) to predict RSSI for
            
        Returns:
            Tuple of (predicted_rssi, uncertainty)
        """
        if len(self.rssi_sequence) < 2:
            return 0.0, 1.0  # High uncertainty when insufficient data
        
        # Prepare training data
        X = np.array(self.position_sequence)
        y = np.array(self.rssi_sequence)
        
        # Fit GP model and make prediction
        self.gp.fit(X, y)
        next_rssi, std = self.gp.predict(np.array([current_position]), return_std=True)
        
        return float(next_rssi[0]), float(std[0])
    
    def calculate_sift_uncertainty(self, rssi_sequence: List[float]) -> float:
        """
        Calculate uncertainty using the SIFT algorithm.
        
        SIFT (Selects Informative data for Fine-Tuning) analyzes the sequence
        of RSSI measurements to estimate prediction uncertainty:
        
        1. Converts sequence to tensor format
        2. Calculates sequence statistics (mean, std)
        3. Applies kernel-based uncertainty estimation
        4. Uses the formula from the research paper:
           uncertainty = sqrt(k(S,S) - k_X^T(S)(K_X + λκI_n)^(-1)k_X(S))
        
        Args:
            rssi_sequence: Sequence of RSSI measurements
            
        Returns:
            Uncertainty score between 0 and 1
        """
        if len(rssi_sequence) < self.sequence_length:
            return 1.0  # Maximum uncertainty when insufficient data
        
        # Convert to PyTorch tensor for calculations
        sequence_tensor = torch.tensor(rssi_sequence, dtype=torch.float32)
        
        # Calculate sequence statistics
        mean = torch.mean(sequence_tensor)
        std = torch.std(sequence_tensor)
        
        # Define kernel function (RBF kernel)
        k = lambda x, y: torch.exp(-torch.norm(x - y) ** 2 / (2 * std ** 2))
        
        # Build kernel matrix
        K = torch.zeros((len(sequence_tensor), len(sequence_tensor)))
        for i in range(len(sequence_tensor)):
            for j in range(len(sequence_tensor)):
                K[i, j] = k(sequence_tensor[i], sequence_tensor[j])
        
        # Calculate SIFT uncertainty score
        lambda_reg = 0.1  # Regularization parameter
        uncertainty = torch.sqrt(k(sequence_tensor[-1], sequence_tensor[-1]) - 
                               torch.matmul(K[-1, :-1], 
                                          torch.linalg.solve(K[:-1, :-1] + lambda_reg * torch.eye(len(sequence_tensor)-1),
                                                          K[:-1, -1])))
        
        return float(uncertainty)
    
    def select_action(self, current_position: Tuple[int, int], 
                     current_rssi: float) -> Tuple[str, float]:
        """
        Select the optimal action based on RSSI predictions, uncertainty, and exploration.
        
        The selection process:
        1. Evaluates each possible action
        2. Predicts RSSI value and uncertainty for resulting position
        3. Calculates score combining prediction, uncertainty, and exploration bonus
        4. Selects action with highest score
        
        Scoring formula: 
        score = (predicted_rssi * (1 - uncertainty)) + exploration_bonus
        
        Args:
            current_position: Current robot position (x, y)
            current_rssi: Current RSSI measurement
            
        Returns:
            Tuple of (selected_action, confidence_score)
        """
        action_scores = {}
        
        # Update visit count for current position
        self.visited_positions[current_position] = self.visited_positions.get(current_position, 0) + 1
        
        # Evaluate each possible action
        for action in self.actions:
            next_position = self._get_next_position(current_position, action)
            
            # Skip invalid positions (outside grid)
            if not self._is_valid_position(next_position):
                continue
            
            # Predict RSSI and uncertainty for this action
            predicted_rssi, prediction_std = self.predict_next_rssi(next_position)
            uncertainty = self.calculate_sift_uncertainty(self.rssi_sequence + [predicted_rssi])
            
            # Calculate exploration bonus (higher for less visited positions)
            visit_count = self.visited_positions.get(next_position, 0)
            exploration_bonus = self.exploration_weight * (1.0 / (1.0 + visit_count))
            
            # Apply visit decay to encourage revisiting old positions after some time
            if visit_count > 0:
                exploration_bonus *= (self.visit_decay ** visit_count)
            
            # Calculate combined score
            rssi_score = predicted_rssi * (1 - uncertainty)
            total_score = rssi_score + exploration_bonus
            
            action_scores[action] = total_score
        
        if not action_scores:
            return 'straight', 0.0
            
        # Select action with highest score
        best_action = max(action_scores.items(), key=lambda x: x[1])
        return best_action[0], best_action[1]
    
    def _get_next_position(self, current_position: Tuple[int, int], 
                          action: str) -> Tuple[int, int]:
        """
        Calculate the next position based on the selected action.
        
        Args:
            current_position: Current position (x, y)
            action: Selected action (left, right, straight, back)
            
        Returns:
            Next position (x, y)
        """
        x, y = current_position
        if action == 'left':
            return (x, y - 1)
        elif action == 'right':
            return (x, y + 1)
        elif action == 'straight':
            return (x - 1, y)
        elif action == 'back':
            return (x + 1, y)
        return current_position
    
    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is valid within the grid boundaries.
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            True if position is valid, False otherwise
        """
        x, y = position
        return (0 <= x < self.grid_size and 
                0 <= y < self.grid_size)
    
    def get_confidence_score(self) -> float:
        """
        Calculate overall confidence score based on recent measurements.
        
        The confidence score is the inverse of uncertainty:
        confidence = 1 - min(uncertainty, 1.0)
        
        Returns:
            Confidence score between 0 and 1
        """
        if len(self.rssi_sequence) < self.sequence_length:
            return 0.0  # No confidence when insufficient data
        
        uncertainty = self.calculate_sift_uncertainty(self.rssi_sequence)
        return 1.0 - min(uncertainty, 1.0)  # Convert uncertainty to confidence
