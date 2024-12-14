import numpy as np
from scipy.stats import rice

class RSSIGridManager:
    def __init__(self):
        # Original RSSI values (base grid)
        self.base_grid = [
            [-8.738930154245397, -7.325638626280702, -6.186205103212336, -5.728630197605584, -6.186205103212336, -7.325638626280702, -8.738930154245397, -10.16560518993271, -11.500994273634886, -12.718330240965772, -13.820485038841705],
            [-7.325638626280702, -5.21710497313177, -3.1759051465725197, -2.2068050164919573, -3.1759051465725197, -5.21710497313177, -7.325638626280702, -9.196505059852145, -10.810185082201892, -12.20680501649196, -13.428963799220222],
            [-6.186205103212336, -3.1759051465725197, 0.0, 0.0, 0.0, -3.1759051465725197, -6.186205103212336, -8.490694316995073, -10.335938582920514, -11.868222343882286, -13.175905146572523],
            [-5.728630197605584, -2.2068050164919573, 0.0, 0.0, 0.0, -2.2068050164919573, -5.728630197605584, -8.227404929771582, -10.16560518993271, -11.74923011088521, -13.08816590349747],
            [-6.186205103212336, -3.1759051465725197, 0.0, 0.0, 0.0, -3.1759051465725197, -6.186205103212336, -8.490694316995073, -10.335938582920514, -11.868222343882286, -13.175905146572523],
            [-7.325638626280702, -5.21710497313177, -3.1759051465725197, -2.2068050164919573, -3.1759051465725197, -5.21710497313177, -7.325638626280702, -9.196505059852145, -10.810185082201892, -12.20680501649196, -13.428963799220222],
            [-8.738930154245397, -7.325638626280702, -6.186205103212336, -5.728630197605584, -6.186205103212336, -7.325638626280702, -8.738930154245397, -10.16560518993271, -11.500994273634886, -12.718330240965772, -13.820485038841705],
            [-10.16560518993271, -9.196505059852145, -8.490694316995073, -8.227404929771582, -8.490694316995073, -9.196505059852145, -10.16560518993271, -11.237704886411395, -12.31404367040969, -13.346238539560327, -14.315338669640889],
            [-11.500994273634886, -10.810185082201892, -10.335938582920514, -10.16560518993271, -10.335938582920514, -10.810185082201892, -11.500994273634886, -12.31404367040969, -13.175905146572523, -14.039503453320007, -14.878522300522095],
            [-12.718330240965772, -12.20680501649196, -11.868222343882286, -11.74923011088521, -11.868222343882286, -12.20680501649196, -12.718330240965772, -13.346238539560327, -14.039503453320007, -14.759530067525015, -15.48039436035526],
            [-13.820485038841705, -13.428963799220222, -13.175905146572523, -13.08816590349747, -13.175905146572523, -13.428963799220222, -13.820485038841705, -14.315338669640889, -14.878522300522095, -15.48039436035526, -16.09846586013728],
        ]
        
        # Initialize all grid versions
        self.current_grid = self.base_grid
        self.current_type = 'base'
        self.initialize_grids()
        
    def initialize_grids(self):
        """Initialize all grid versions"""
        self.grids = {
            'base': self.base_grid,
            'gaussian': self.add_gaussian_noise(),
            'rayleigh': self.apply_rayleigh_fading(),
            'rician': self.apply_rician_fading(),
            'path_loss': self.apply_path_loss()
        }
    
    def add_gaussian_noise(self, mu=0, sigma=2.5):
        """
        Noise-Based Augmentation with Gaussian noise
        σ = 2.5 dBm (empirically calibrated)
        """
        grid = np.array(self.base_grid)
        noise = np.random.normal(mu, sigma, size=grid.shape)
        return self._preserve_targets(grid + noise)

    def apply_rayleigh_fading(self):
        """
        Channel Condition Simulation - Rayleigh Fading
        sfaded = soriginal × h, where h ~ R(0, 1)
        """
        grid = np.array(self.base_grid)
        h = np.random.rayleigh(scale=1.0, size=grid.shape)
        return self._preserve_targets(grid * h)

    def apply_rician_fading(self, K=1, omega=1):
        """
        Channel Condition Simulation - Rician Fading
        K: Rician K-factor
        Ω: mean power
        """
        grid = np.array(self.base_grid)
        x = np.random.normal(0, 1, size=grid.shape)
        y = np.random.normal(0, 1, size=grid.shape)
        
        rician_factor = np.sqrt((K * omega)/(K + 1)) + np.sqrt(omega/(2*(K + 1))) * (x + y)
        return self._preserve_targets(grid * rician_factor)

    def apply_path_loss(self, d0=1, alpha=2, pt=0, gt=0, gr=0):
        """
        Path Loss Model Integration
        Pr = Pt + Gt + Gr - Lp(d0) - 10α*log(d/d0) + Xσ
        """
        grid = np.array(self.base_grid)
        rows, cols = grid.shape
        
        center_x, center_y = rows//2, cols//2
        y, x = np.ogrid[-center_x:rows-center_x, -center_y:cols-center_y]
        distances = np.sqrt(x*x + y*y)
        
        with np.errstate(divide='ignore'):
            path_loss = pt + gt + gr - (10 * alpha * np.log10(np.maximum(distances/d0, 0.1)))
        
        shadow_fading = np.random.normal(0, 2, size=grid.shape)
        return self._preserve_targets(grid + path_loss + shadow_fading)
    
    def _preserve_targets(self, grid):
        """Ensure target positions (0.0) are preserved and non-target values remain negative"""
        grid = np.array(grid)
        base = np.array(self.base_grid)
        grid = np.where(base == 0, 0, grid)  # Preserve target positions
        grid = np.where(grid > 0, -grid, grid)  # Ensure non-target values are negative
        return grid.tolist()
    
    def switch_grid(self, grid_type):
        """Switch to a different RSSI grid type"""
        if grid_type in self.grids:
            self.current_grid = self.grids[grid_type]
            self.current_type = grid_type
            return True
        return False
    
    def get_current_grid(self):
        """Get the current RSSI grid"""
        return self.current_grid
    
    def refresh_grid(self, grid_type):
        """Refresh a specific grid type with new random variations"""
        if grid_type == 'base':
            return  # Base grid never changes
            
        if grid_type == 'gaussian':
            self.grids[grid_type] = self.add_gaussian_noise()
        elif grid_type == 'rayleigh':
            self.grids[grid_type] = self.apply_rayleigh_fading()
        elif grid_type == 'rician':
            self.grids[grid_type] = self.apply_rician_fading()
        elif grid_type == 'path_loss':
            self.grids[grid_type] = self.apply_path_loss()
        
        # Update current grid if we're refreshing the current type
        if grid_type == self.current_type:
            self.current_grid = self.grids[grid_type]
