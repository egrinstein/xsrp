# Standard library imports
import sys
from pathlib import Path

# Third-party imports - optional GUI dependencies
try:
    import numpy as np
    import pyaudio
    import yaml
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from PyQt5.QtCore import QTimer, Qt
    from PyQt5.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
        QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressDialog, QPushButton, QSlider,
        QVBoxLayout, QWidget
    )
except ImportError as e:
    raise ImportError(
        "GUI dependencies are not installed. Install them with: pip install xsrp[gui]\n"
        f"Original error: {e}"
    ) from e

# Local imports
from xsrp.calibrate import (
    record_ambient_noise, compute_noise_floor_map,
    save_ambient_noise, load_ambient_noise,
    calculate_aliasing_limit
)
from xsrp.xsrp import XSrp
from xsrp.grids import UniformSphericalGrid
from xsrp.streaming import StreamingSrp
from xsrp.tracking import ExponentialSmoothingTracker
from visualization.polar import plot_polar_srp_map

# Constants
DEFAULT_SAMPLING_RATE = 16000
DEFAULT_FRAME_SIZE = 1024
DEFAULT_SMOOTHING_ALPHA = 0.2
DEFAULT_N_AVERAGE_SAMPLES = 1
DEFAULT_N_AZIMUTH_CELLS = 360  # 1 degree resolution
DEFAULT_SHARPENING = 1.0
DEFAULT_FILTER_ENABLED = True
DEFAULT_FILTER_LOWCUT = 200.0  # High-pass cutoff in Hz
DEFAULT_FILTER_HIGHCUT = 6000.0  # Low-pass cutoff in Hz
DEFAULT_SRP_MODE = "gcc_phat_freq"
DEFAULT_FREQUENCY_WEIGHTING = None

# UI Constants
WINDOW_TITLE = "Real-time SRP GUI"
WINDOW_X = 100
WINDOW_Y = 100
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
PLOT_FIGSIZE = (8, 8)

# Slider ranges and defaults
SMOOTHING_SLIDER_MIN = 0
SMOOTHING_SLIDER_MAX = 100
SMOOTHING_SLIDER_DEFAULT = 20  # Maps to 0.20
AVERAGING_SLIDER_MIN = 0
AVERAGING_SLIDER_MAX = 20
AVERAGING_SLIDER_DEFAULT = 1
RESOLUTION_SLIDER_MIN = 1  # 1 degree (360 cells)
RESOLUTION_SLIDER_MAX = 18  # 18 degrees (20 cells)
RESOLUTION_SLIDER_DEFAULT = 1
SHARPENING_SLIDER_MIN = 10  # 1.0 (no sharpening)
SHARPENING_SLIDER_MAX = 80  # 8.0 (max sharpening)
SHARPENING_SLIDER_DEFAULT = 10
LOWCUT_SLIDER_MIN = 0
LOWCUT_SLIDER_MAX = 1000
LOWCUT_SLIDER_DEFAULT = 200
HIGHCUT_SLIDER_MIN = 1000
HIGHCUT_SLIDER_MAX = 8000  # Nyquist for 16kHz
HIGHCUT_SLIDER_DEFAULT = 6000

# Calibration defaults
CALIBRATION_DURATION_DEFAULT = 5.0
CALIBRATION_DURATION_MIN = 1.0
CALIBRATION_DURATION_MAX = 60.0

# File paths
DEFAULT_CONFIG_DIR = "docs/mic_config"
DEFAULT_CONFIG_FILE = "respeaker.yaml"
DEFAULT_NOISE_FLOOR_DIR = "temp"
DEFAULT_NOISE_FLOOR_FILE = "srp_noise_floor.h5"

# Radial plot adaptation
RADIAL_MAX_MULTIPLIER = 1.1


class PolarPlotWidget(QWidget):
    """Widget for displaying polar SRP map."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=PLOT_FIGSIZE)
        self.canvas = FigureCanvas(self.figure)
        self.ax = None
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def initialize_plot(self, grid):
        """Initialize the polar plot with grid."""
        if self.ax is None:
            self.ax = self.figure.add_subplot(111, projection='polar')
        self.grid = grid
    
    def update_plot(self, srp_map, tracked_position=None, radial_max=None, show_tracked=True):
        """Update the plot with new SRP map data."""
        if self.ax is None:
            return
        
        self.ax.clear()
        plot_polar_srp_map(self.grid, srp_map, tracked_position, ax=self.ax, colorbar=False, 
                          radial_max=radial_max, show_tracked=show_tracked)
        self.canvas.draw()


class SRPGUI(QMainWindow):
    """Main GUI application for real-time SRP."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(WINDOW_X, WINDOW_Y, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Audio and processing state
        self.audio = None
        self.stream = None
        self.device_channels = 0
        self.is_recording = False
        self.config = None
        self.mic_positions = None
        self.streaming_srp = None
        self.grid = None
        self.radial_max = None  # Maximum radial limit for polar plot
        self.noise_floor = None  # Noise floor SRP map for subtraction
        self.ambient_audio = None # Raw ambient audio data
        self.noise_floor_path = None  # Path to noise floor file
        self.noise_floor_enabled = True  # Whether to subtract noise floor
        
        # Processing parameters
        self.fs = DEFAULT_SAMPLING_RATE
        self.frame_size = DEFAULT_FRAME_SIZE
        self.hop_size = self.frame_size // 2  # 50% overlap
        self.smoothing_alpha = DEFAULT_SMOOTHING_ALPHA
        self.n_average_samples = DEFAULT_N_AVERAGE_SAMPLES
        self.n_azimuth_cells = DEFAULT_N_AZIMUTH_CELLS
        self.sharpening = DEFAULT_SHARPENING
        
        # Filtering parameters
        self.filter_enabled = DEFAULT_FILTER_ENABLED
        self.filter_lowcut = DEFAULT_FILTER_LOWCUT
        self.filter_highcut = DEFAULT_FILTER_HIGHCUT
        
        # SRP mode parameter
        self.srp_mode = DEFAULT_SRP_MODE
        
        # Frequency weighting parameter
        self.frequency_weighting = DEFAULT_FREQUENCY_WEIGHTING
        
        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_audio_frame)
        
        # Initialize UI
        self.init_ui()
        
        # Load default config
        default_config_path = Path(__file__).parent.parent / DEFAULT_CONFIG_DIR / DEFAULT_CONFIG_FILE
        if default_config_path.exists():
            self.load_config(str(default_config_path))
        
        # Try to load noise floor if it exists
        self.load_noise_floor_auto()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel: Controls
        controls_panel = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        # Audio device selection
        device_label = QLabel("Audio Device:")
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        controls_layout.addWidget(device_label)
        controls_layout.addWidget(self.device_combo)
        
        # Config file selection
        config_label = QLabel("Microphone Config:")
        config_layout = QHBoxLayout()
        self.config_label = QLabel("No config loaded")
        config_button = QPushButton("Load Config...")
        config_button.clicked.connect(self.load_config_dialog)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(config_button)
        controls_layout.addWidget(config_label)
        controls_layout.addWidget(QLabel())  # Spacer
        controls_layout.addLayout(config_layout)
        
        # Smoothing control
        smoothing_label = QLabel("Smoothing Speed:")
        smoothing_layout = QHBoxLayout()
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setMinimum(SMOOTHING_SLIDER_MIN)
        self.smoothing_slider.setMaximum(SMOOTHING_SLIDER_MAX)
        self.smoothing_slider.setValue(SMOOTHING_SLIDER_DEFAULT)
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        self.smoothing_value_label = QLabel(f"{DEFAULT_SMOOTHING_ALPHA:.2f}")
        smoothing_layout.addWidget(self.smoothing_slider)
        smoothing_layout.addWidget(self.smoothing_value_label)
        controls_layout.addWidget(smoothing_label)
        controls_layout.addLayout(smoothing_layout)
        
        # Averaging control
        averaging_label = QLabel("Averaging Samples:")
        averaging_layout = QHBoxLayout()
        self.averaging_slider = QSlider(Qt.Horizontal)
        self.averaging_slider.setMinimum(AVERAGING_SLIDER_MIN)
        self.averaging_slider.setMaximum(AVERAGING_SLIDER_MAX)
        self.averaging_slider.setValue(AVERAGING_SLIDER_DEFAULT)
        self.averaging_slider.valueChanged.connect(self.on_averaging_changed)
        self.averaging_slider.sliderReleased.connect(self.on_averaging_released)
        self.averaging_value_label = QLabel(str(DEFAULT_N_AVERAGE_SAMPLES))
        averaging_layout.addWidget(self.averaging_slider)
        averaging_layout.addWidget(self.averaging_value_label)
        controls_layout.addWidget(averaging_label)
        controls_layout.addLayout(averaging_layout)
        
        # Resolution control
        resolution_label = QLabel("Resolution (degrees):")
        resolution_layout = QHBoxLayout()
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setMinimum(RESOLUTION_SLIDER_MIN)
        self.resolution_slider.setMaximum(RESOLUTION_SLIDER_MAX)
        self.resolution_slider.setValue(RESOLUTION_SLIDER_DEFAULT)
        self.resolution_slider.valueChanged.connect(self.on_resolution_changed)
        self.resolution_slider.sliderReleased.connect(self.on_resolution_released)
        self.resolution_value_label = QLabel(f"{360 / DEFAULT_N_AZIMUTH_CELLS:.1f}°")
        resolution_layout.addWidget(self.resolution_slider)
        resolution_layout.addWidget(self.resolution_value_label)
        controls_layout.addWidget(resolution_label)
        controls_layout.addLayout(resolution_layout)

        # Sharpening control
        sharpening_label = QLabel("Sharpening:")
        sharpening_layout = QHBoxLayout()
        self.sharpening_slider = QSlider(Qt.Horizontal)
        self.sharpening_slider.setMinimum(SHARPENING_SLIDER_MIN)
        self.sharpening_slider.setMaximum(SHARPENING_SLIDER_MAX)
        self.sharpening_slider.setValue(SHARPENING_SLIDER_DEFAULT)
        self.sharpening_slider.valueChanged.connect(self.on_sharpening_changed)
        self.sharpening_slider.sliderReleased.connect(self.on_sharpening_released)
        self.sharpening_value_label = QLabel(f"{DEFAULT_SHARPENING:.1f}")
        sharpening_layout.addWidget(self.sharpening_slider)
        sharpening_layout.addWidget(self.sharpening_value_label)
        controls_layout.addWidget(sharpening_label)
        controls_layout.addLayout(sharpening_layout)
        
        # Filter controls
        filter_group = QGroupBox("Bandpass Filter")
        filter_layout = QVBoxLayout()
        
        # Enable filter checkbox
        self.filter_enabled_checkbox = QCheckBox("Enable Filtering")
        self.filter_enabled_checkbox.setChecked(DEFAULT_FILTER_ENABLED)
        self.filter_enabled_checkbox.stateChanged.connect(self.on_filter_enabled_changed)
        filter_layout.addWidget(self.filter_enabled_checkbox)
        
        # Low cutoff (high-pass)
        lowcut_label = QLabel("High-pass (Hz):")
        lowcut_layout = QHBoxLayout()
        self.lowcut_slider = QSlider(Qt.Horizontal)
        self.lowcut_slider.setMinimum(LOWCUT_SLIDER_MIN)
        self.lowcut_slider.setMaximum(LOWCUT_SLIDER_MAX)
        self.lowcut_slider.setValue(LOWCUT_SLIDER_DEFAULT)
        self.lowcut_slider.valueChanged.connect(self.on_lowcut_changed)
        self.lowcut_slider.sliderReleased.connect(self.on_lowcut_released)
        self.lowcut_value_label = QLabel(str(int(DEFAULT_FILTER_LOWCUT)))
        lowcut_layout.addWidget(self.lowcut_slider)
        lowcut_layout.addWidget(self.lowcut_value_label)
        filter_layout.addWidget(lowcut_label)
        filter_layout.addLayout(lowcut_layout)
        
        # High cutoff (low-pass)
        highcut_label = QLabel("Low-pass (Hz):")
        highcut_layout = QHBoxLayout()
        self.highcut_slider = QSlider(Qt.Horizontal)
        self.highcut_slider.setMinimum(HIGHCUT_SLIDER_MIN)
        self.highcut_slider.setMaximum(HIGHCUT_SLIDER_MAX)
        self.highcut_slider.setValue(HIGHCUT_SLIDER_DEFAULT)
        self.highcut_slider.valueChanged.connect(self.on_highcut_changed)
        self.highcut_slider.sliderReleased.connect(self.on_highcut_released)
        self.highcut_value_label = QLabel(str(int(DEFAULT_FILTER_HIGHCUT)))
        highcut_layout.addWidget(self.highcut_slider)
        highcut_layout.addWidget(self.highcut_value_label)
        filter_layout.addWidget(highcut_label)
        filter_layout.addLayout(highcut_layout)
        
        filter_group.setLayout(filter_layout)
        controls_layout.addWidget(filter_group)
        
        # SRP mode selection
        mode_label = QLabel("SRP Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Time Domain", "gcc_phat_time")
        self.mode_combo.addItem("Frequency Domain", "gcc_phat_freq")
        self.mode_combo.setCurrentIndex(1)  # Default to frequency domain
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(mode_label)
        controls_layout.addWidget(self.mode_combo)
        
        # Frequency weighting selection
        frequency_weighting_label = QLabel("Frequency Weighting:")
        self.frequency_weighting_combo = QComboBox()
        self.frequency_weighting_combo.addItem("None", None)
        self.frequency_weighting_combo.addItem("Coherence", "coherence")
        self.frequency_weighting_combo.addItem("Sparsity", "sparsity")
        self.frequency_weighting_combo.addItem("PAR", "par")
        self.frequency_weighting_combo.setCurrentIndex(0)  # Default to None
        self.frequency_weighting_combo.setEnabled(True)  # Enabled by default (freq mode is default)
        self.frequency_weighting_combo.currentIndexChanged.connect(self.on_frequency_weighting_changed)
        controls_layout.addWidget(frequency_weighting_label)
        controls_layout.addWidget(self.frequency_weighting_combo)
        
        # Show tracked position toggle
        self.show_tracked_checkbox = QCheckBox("Show Tracked Position")
        self.show_tracked_checkbox.setChecked(True)  # Default to showing
        self.show_tracked_checkbox.stateChanged.connect(self.on_show_tracked_changed)
        controls_layout.addWidget(self.show_tracked_checkbox)
        
        # Noise floor calibration
        noise_floor_label = QLabel("Noise Floor:")
        noise_floor_layout = QHBoxLayout()
        self.noise_floor_status_label = QLabel("Not loaded")
        calibrate_button = QPushButton("Calibrate...")
        calibrate_button.clicked.connect(self.calibrate_noise_floor)
        load_noise_floor_button = QPushButton("Load...")
        load_noise_floor_button.clicked.connect(self.load_noise_floor_dialog)
        noise_floor_layout.addWidget(self.noise_floor_status_label)
        noise_floor_layout.addWidget(calibrate_button)
        noise_floor_layout.addWidget(load_noise_floor_button)
        controls_layout.addWidget(noise_floor_label)
        controls_layout.addLayout(noise_floor_layout)
        
        # Enable noise floor subtraction checkbox
        self.noise_floor_enabled_checkbox = QCheckBox("Enable Noise Floor Subtraction")
        self.noise_floor_enabled_checkbox.setChecked(True)  # Default to enabled
        self.noise_floor_enabled_checkbox.stateChanged.connect(self.on_noise_floor_enabled_changed)
        controls_layout.addWidget(self.noise_floor_enabled_checkbox)
        
        # Start/Stop button
        self.start_stop_button = QPushButton("Start Recording")
        self.start_stop_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.start_stop_button)
        
        # Status display
        self.status_label = QLabel("Status: Ready")
        controls_layout.addWidget(self.status_label)
        
        controls_layout.addStretch()
        controls_panel.setLayout(controls_layout)
        
        # Right panel: Polar plot
        self.polar_plot = PolarPlotWidget()
        
        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(self.polar_plot, 3)
        
        # Populate audio devices
        self.refresh_audio_devices()
    
    def refresh_audio_devices(self):
        """Refresh the list of available audio input devices."""
        self.device_combo.clear()
        
        if self.audio is None:
            try:
                self.audio = pyaudio.PyAudio()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize PyAudio: {e}")
                return
        
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                name = info['name']
                channels = info['maxInputChannels']
                self.device_combo.addItem(f"{name} ({channels} ch)", i)
    
    def on_device_changed(self):
        """Handle audio device selection change."""
        if self.is_recording:
            self.stop_recording()
    
    def on_smoothing_changed(self, value):
        """Handle smoothing slider change."""
        self.smoothing_alpha = value / 100.0
        self.smoothing_value_label.setText(f"{self.smoothing_alpha:.2f}")
        # Update tracker if recording
        if self.is_recording and self.streaming_srp is not None:
            # Update tracker alpha and reset state
            self.streaming_srp.tracker.alpha = self.smoothing_alpha
            self.streaming_srp.tracker.reset()
    
    def on_averaging_changed(self, value):
        """Handle averaging slider change (UI update only)."""
        self.n_average_samples = value
        self.averaging_value_label.setText(str(value))
    
    def on_averaging_released(self):
        """Handle averaging slider release (Logic update)."""
        if self.is_recording and self.streaming_srp is not None:
            # Clear noise floor immediately to avoid shape mismatches
            self.noise_floor = None
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                QMessageBox.warning(self, "Error", "Failed to reinitialize SRP processor")
            else:
                # Averaging affects SRP computation, so recompute noise floor
                if self.should_recompute_noise_floor():
                    self.recompute_noise_floor(show_progress=True, reason="averaging changed")

    def on_resolution_changed(self, value):
        """Handle resolution slider change (UI update only)."""
        degrees_per_cell = float(value)
        self.n_azimuth_cells = int(360 / degrees_per_cell)
        self.resolution_value_label.setText(f"{degrees_per_cell:.1f}°")
    
    def on_resolution_released(self):
        """Handle resolution slider release (Logic update)."""
        if self.is_recording and self.streaming_srp is not None:
            # Clear noise floor immediately to avoid shape mismatches (grid size changed)
            self.noise_floor = None
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                QMessageBox.warning(self, "Error", "Failed to reinitialize SRP processor")
            else:
                # Resolution affects grid, so recompute noise floor
                if self.should_recompute_noise_floor():
                    self.recompute_noise_floor(show_progress=True, reason="resolution changed")

    def on_sharpening_changed(self, value):
        """Handle sharpening slider change (UI update only)."""
        self.sharpening = value / 10.0
        self.sharpening_value_label.setText(f"{self.sharpening:.1f}")
    
    def on_sharpening_released(self):
        """Handle sharpening slider release (Logic update)."""
        # Update processor if it exists
        if self.streaming_srp is not None:
            if hasattr(self.streaming_srp.srp_processor, 'sharpening'):
                self.streaming_srp.srp_processor.sharpening = self.sharpening
        else:
            # If processor doesn't exist, initialize it
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                return
        
        # Recompute noise floor since sharpening affects it
        # Clear first to avoid any issues
        if self.should_recompute_noise_floor():
            self.noise_floor = None
            self.recompute_noise_floor(show_progress=True, reason="sharpening changed")

    def on_show_tracked_changed(self, state):
        """Handle show tracked position checkbox change."""
        # The next frame update will use the new setting
        pass
    
    def on_noise_floor_enabled_changed(self, state):
        """Handle noise floor enabled checkbox change."""
        self.noise_floor_enabled = state == Qt.Checked
    
    def on_filter_enabled_changed(self, state):
        """Handle filter enabled checkbox change."""
        self.filter_enabled = state == Qt.Checked
        if self.streaming_srp is not None:
            self.streaming_srp.filter_enabled = self.filter_enabled
            # Recompute noise floor since filtering affects input
            if self.should_recompute_noise_floor():
                self.noise_floor = None
                self.recompute_noise_floor(show_progress=True, reason="filter enabled changed")
    
    def on_lowcut_changed(self, value):
        """Handle lowcut slider change (UI update only)."""
        self.filter_lowcut = float(value)
        self.lowcut_value_label.setText(str(value))
    
    def on_lowcut_released(self):
        """Handle lowcut slider release (Logic update)."""
        if self.streaming_srp is not None:
            self.streaming_srp.filter_lowcut = self.filter_lowcut
            # Recompute noise floor since filter cutoff affects input
            if self.should_recompute_noise_floor():
                self.noise_floor = None
                self.recompute_noise_floor(show_progress=True, reason="high-pass cutoff changed")

    def on_highcut_changed(self, value):
        """Handle highcut slider change (UI update only)."""
        self.filter_highcut = float(value)
        self.highcut_value_label.setText(str(value))

    def on_highcut_released(self):
        """Handle highcut slider release (Logic update)."""
        if self.streaming_srp is not None:
            self.streaming_srp.filter_highcut = self.filter_highcut
            
            # Sync frequency cutoff with SRP processor to avoid PHAT noise in stop-band
            if hasattr(self.streaming_srp.srp_processor, 'freq_cutoff'):
                # Convert cutoff Hz to bin index
                fs = self.streaming_srp.srp_processor.fs
                n_dft = self.streaming_srp.srp_processor.n_dft_bins
                new_cutoff = int(self.filter_highcut * n_dft / fs)
                self.streaming_srp.srp_processor.freq_cutoff = new_cutoff
            
            # Recompute noise floor since filter cutoff affects input
            if self.should_recompute_noise_floor():
                self.noise_floor = None
                self.recompute_noise_floor(show_progress=True, reason="low-pass cutoff changed")
    
    def on_mode_changed(self, index):
        """Handle SRP mode selection change."""
        self.srp_mode = self.mode_combo.currentData()
        # Enable/disable frequency weighting based on mode
        is_freq_mode = self.srp_mode == "gcc_phat_freq"
        self.frequency_weighting_combo.setEnabled(is_freq_mode)
        # If switching away from freq mode, disable frequency weighting
        if not is_freq_mode:
            self.frequency_weighting = None
            self.frequency_weighting_combo.setCurrentIndex(0)
        # Reinitialize SRP processor if recording
        if self.is_recording and self.streaming_srp is not None:
            # Clear noise floor immediately to avoid shape mismatches
            self.noise_floor = None
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                QMessageBox.warning(self, "Error", "Failed to reinitialize SRP processor")
            else:
                # Mode affects SRP computation, so recompute noise floor
                if self.should_recompute_noise_floor():
                    self.recompute_noise_floor(show_progress=True, reason="SRP mode changed")
    
    def on_frequency_weighting_changed(self, index):
        """Handle frequency weighting selection change."""
        self.frequency_weighting = self.frequency_weighting_combo.currentData()
        # Reinitialize SRP processor if recording (since frequency_weighting is set at initialization)
        if self.is_recording and self.streaming_srp is not None:
            # Clear noise floor immediately to avoid shape mismatches
            self.noise_floor = None
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                QMessageBox.warning(self, "Error", "Failed to reinitialize SRP processor")
            else:
                # Frequency weighting affects SRP computation, so recompute noise floor
                if self.should_recompute_noise_floor():
                    self.recompute_noise_floor(show_progress=True, reason="frequency weighting changed")
    
    def get_noise_floor_path(self) -> Path:
        """Get the default path for noise floor file."""
        return Path(__file__).parent.parent / DEFAULT_NOISE_FLOOR_DIR / DEFAULT_NOISE_FLOOR_FILE
    
    def load_noise_floor_auto(self):
        """Automatically load noise floor if it exists."""
        noise_floor_path = self.get_noise_floor_path()
        if noise_floor_path.exists():
            try:
                self.ambient_audio, fs = load_ambient_noise(str(noise_floor_path))
                if self.ambient_audio is not None:
                    self.noise_floor_path = noise_floor_path
                    self.noise_floor_status_label.setText("Loaded (Audio)")
                    # Compute map if SRP is ready, otherwise wait
                    if self.streaming_srp is not None:
                        self.recompute_noise_floor(show_progress=False, reason="auto-load on startup")
                else:
                    self.noise_floor = None
                    self.noise_floor_status_label.setText("Invalid format")
            except Exception as e:
                print(f"Failed to load noise floor: {e}")
                self.noise_floor = None
                self.noise_floor_status_label.setText("Load failed")
        else:
            self.noise_floor = None
            self.noise_floor_status_label.setText("Not found")
    
    def load_noise_floor_dialog(self):
        """Open dialog to load noise floor file."""
        default_path = Path(__file__).parent.parent / DEFAULT_NOISE_FLOOR_DIR
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Noise Floor",
            str(default_path),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if file_path:
            try:
                self.ambient_audio, fs = load_ambient_noise(file_path)
                if self.ambient_audio is not None:
                    self.noise_floor_path = Path(file_path)
                    self.noise_floor_status_label.setText(f"Loaded: {Path(file_path).name}")
                    # Compute map if SRP is ready, otherwise wait
                    if self.streaming_srp is not None:
                        self.recompute_noise_floor(show_progress=True, reason="noise floor loaded")
                else:
                     QMessageBox.warning(self, "Invalid File", "File does not contain valid ambient audio data.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load noise floor: {e}")
                self.noise_floor = None
                self.noise_floor_status_label.setText("Load failed")
    
    def calibrate_noise_floor(self):
        """Calibrate noise floor by recording in silent environment."""
        if not self.validate_config():
            return
        
        if self.is_recording:
            QMessageBox.warning(self, "Error", "Please stop recording before calibrating")
            return
        
        # Ask for duration
        duration, ok = QInputDialog.getDouble(
            self,
            "Calibration Duration",
            "Enter duration in seconds:",
            CALIBRATION_DURATION_DEFAULT,
            CALIBRATION_DURATION_MIN,
            CALIBRATION_DURATION_MAX,
            1  # Decimals
        )
        
        if not ok:
            return
        
        device_index = self.device_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, "Error", "Please select an audio device")
            return
        
        # Show progress dialog
        progress_dialog = QMessageBox(self)
        progress_dialog.setWindowTitle("Calibrating Noise Floor")
        progress_dialog.setText(f"Recording ambient noise for {duration} seconds...\nPlease ensure the environment is silent.")
        progress_dialog.setStandardButtons(QMessageBox.NoButton)
        progress_dialog.setModal(False)
        progress_dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Record raw ambient noise
            self.ambient_audio = record_ambient_noise(
                fs=self.fs,
                duration_seconds=duration,
                device_index=device_index,
                chunk_size=self.frame_size,
                progress_callback=lambda c, t: QApplication.processEvents()
            )
            
            # Save raw audio
            noise_floor_path = self.get_noise_floor_path()
            save_ambient_noise(self.ambient_audio, str(noise_floor_path), self.fs)
            self.noise_floor_path = noise_floor_path
            
            # Compute the actual noise floor map using current settings
            # Show progress since recording is done
            self.recompute_noise_floor(show_progress=True, reason="calibration completed")
            
            # Close progress dialog properly
            progress_dialog.hide()
            progress_dialog.deleteLater()
            QApplication.processEvents()
            
            QMessageBox.information(self, "Success", f"Noise floor calibrated and saved to {noise_floor_path.name}")
            
        except Exception as e:
            # Close progress dialog properly
            progress_dialog.hide()
            progress_dialog.deleteLater()
            QApplication.processEvents()
            
            QMessageBox.critical(self, "Error", f"Failed to calibrate noise floor: {e}")
            self.ambient_audio = None
            self.noise_floor = None
            self.noise_floor_status_label.setText("Calibration failed")
    
    def should_recompute_noise_floor(self) -> bool:
        """Check if noise floor should be recomputed based on current state.
        
        Returns
        -------
        bool
            True if ambient audio is loaded and noise floor subtraction is enabled
        """
        return (self.ambient_audio is not None and 
                self.noise_floor_enabled and
                self.streaming_srp is not None)
    
    def recompute_noise_floor(self, show_progress=True, reason="settings changed"):
        """Recompute noise floor map from ambient audio using current SRP settings.
        
        This is a modular method that provides consistent notification interface
        for all noise floor recomputation scenarios.
        
        Parameters
        ----------
        show_progress : bool
            Whether to show a progress dialog during computation
        reason : str
            Reason for recomputation (for user notification)
        """
        if self.ambient_audio is None:
            return

        # Ensure SRP processor is initialized
        if self.streaming_srp is None:
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                print("Failed to initialize SRP processor for noise floor computation")
                return

        # Reset noise floor to prevent shape mismatch errors while recomputing
        self.noise_floor = None

        # Calculate number of frames for progress estimation
        n_samples, n_channels = self.ambient_audio.shape
        n_frames = (n_samples - self.frame_size) // self.hop_size + 1
        
        if n_frames <= 0:
            QMessageBox.warning(self, "Error", "Audio data too short for noise floor computation")
            return

        # Show progress dialog if requested
        progress_dialog = None
        if show_progress:
            progress_dialog = QProgressDialog(
                f"Recomputing noise floor map...\nReason: {reason}",
                "Cancel",
                0,
                n_frames,
                self
            )
            progress_dialog.setWindowTitle("Recomputing Noise Floor")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)  # Show immediately
            progress_dialog.setValue(0)
            QApplication.processEvents()

        try:
            # Progress callback for compute_noise_floor_map
            cancelled = [False]  # Use list to allow modification in nested function
            
            def progress_callback(current, total):
                if show_progress and progress_dialog:
                    if progress_dialog.wasCanceled():
                        cancelled[0] = True
                        return False  # Signal to stop
                    progress_pct = int(100 * current / total) if total > 0 else 0
                    progress_dialog.setValue(current)
                    progress_dialog.setLabelText(
                        f"Recomputing noise floor map...\n"
                        f"Reason: {reason}\n"
                        f"Processing frame {current}/{total} ({progress_pct}%)"
                    )
                    QApplication.processEvents()
                return True  # Continue processing
            
            # Compute noise floor with progress updates
            self.noise_floor = compute_noise_floor_map(
                audio_data=self.ambient_audio,
                srp_processor=self.streaming_srp.srp_processor,
                frame_size=self.frame_size,
                hop_size=self.hop_size,
                fs=self.fs,
                filter_enabled=self.filter_enabled,
                filter_lowcut=self.filter_lowcut,
                filter_highcut=self.filter_highcut,
                ignore_channels=self.config.get('ignore_channels', []) if self.config else [],
                progress_callback=progress_callback
            )
            
            # Check if cancelled
            if cancelled[0] or (show_progress and progress_dialog and progress_dialog.wasCanceled()):
                self.noise_floor = None
                self.noise_floor_status_label.setText("Cancelled")
            else:
                self.noise_floor_status_label.setText("Loaded & Computed")
            
        except Exception as e:
            print(f"Error recomputing noise floor: {e}")
            self.noise_floor = None
            self.noise_floor_status_label.setText("Computation failed")
            if show_progress:
                QMessageBox.warning(self, "Error", f"Failed to recompute noise floor: {e}")
        finally:
            if progress_dialog is not None:
                progress_dialog.close()
                QApplication.processEvents()
    
    def load_config_dialog(self):
        """Open dialog to load microphone configuration file."""
        default_path = Path(__file__).parent.parent / DEFAULT_CONFIG_DIR
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Microphone Configuration",
            str(default_path),
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if file_path:
            self.load_config(file_path)
    
    def load_config(self, config_path: str):
        """Load microphone configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate config structure
            if 'microphones' not in self.config:
                raise ValueError("Config must contain 'microphones' field")
            
            # Extract microphone positions
            radius_m = self.config.get('radius_mm', 35) / 1000.0  # Convert mm to meters
            mic_angles = [mic['angle_deg'] for mic in self.config['microphones']]
            
            # Convert to cartesian coordinates (2D, assuming z=0 for circular array)
            self.mic_positions = np.array([
                [radius_m * np.cos(np.radians(angle)), radius_m * np.sin(np.radians(angle)), 0.0]
                for angle in mic_angles
            ])
            
            self.config_label.setText(f"Config: {Path(config_path).name}")
            
            # Check for spatial aliasing and adjust filter
            status_msg = f"Status: Config loaded ({len(mic_angles)} microphones)"
            
            suggested_limit = calculate_aliasing_limit(self.mic_positions)
            
            if suggested_limit is not None:
                # Clamp to slider range
                suggested_limit = max(HIGHCUT_SLIDER_MIN, min(suggested_limit, HIGHCUT_SLIDER_MAX))
                
                # If current setting is too high, lower it
                if suggested_limit < self.highcut_slider.value():
                    self.highcut_slider.setValue(suggested_limit)
                    status_msg = f"Status: Filter set to {suggested_limit}Hz to prevent spatial aliasing"
            
            self.status_label.setText(status_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {e}")
            self.config = None
            self.mic_positions = None
    
    def validate_config(self) -> bool:
        """Validate that config matches selected audio device."""
        if self.config is None:
            QMessageBox.warning(self, "Error", "No microphone configuration loaded")
            return False
        
        if self.device_combo.currentIndex() < 0:
            QMessageBox.warning(self, "Error", "No audio device selected")
            return False
        
        device_index = self.device_combo.currentData()
        device_info = self.audio.get_device_info_by_index(device_index)
        device_channels = device_info['maxInputChannels']
        
        ignore_channels = self.config.get('ignore_channels', [])
        expected_mic_count = len(self.config['microphones'])
        expected_total_channels = expected_mic_count + len(ignore_channels)
        
        if device_channels != expected_total_channels:
            QMessageBox.warning(
                self,
                "Channel Mismatch",
                f"Device has {device_channels} channels, but config expects {expected_total_channels} "
                f"({expected_mic_count} microphones + {len(ignore_channels)} ignored channels)"
            )
            return False
        
        return True
    
    def initialize_srp_processor(self, suppress_noise_floor_recompute=False):
        """Initialize the SRP processor with current configuration.
        
        Parameters
        ----------
        suppress_noise_floor_recompute : bool
            If True, skip automatic noise floor recomputation (used when
            recomputation will be triggered separately)
        """
        if self.mic_positions is None:
            return False
        
        # Create DOA grid (azimuth only)
        self.grid = UniformSphericalGrid(self.n_azimuth_cells)
        
        # For 1D DOA grid, use 2D mic positions (x, y only)
        # The grid positions are 2D, so mic positions must also be 2D
        mic_positions_2d = self.mic_positions[:, :2]  # Take only x, y coordinates
        
        # Create SRP processor with volumetric approach (n_average_samples > 1)
        # Volumetric SRP averages over N-closest correlation values for better robustness
        # Use selected mode (time or frequency domain)
        # Frequency weighting only applies in frequency domain mode
        frequency_weighting = self.frequency_weighting if self.srp_mode == "gcc_phat_freq" else None
        
        srp_processor = XSrp(
            fs=self.fs,
            grid_type="doa_1D",
            n_grid_cells=self.n_azimuth_cells,
            mic_positions=mic_positions_2d,
            mode=self.srp_mode,
            interpolation=False,  # Disabled as requested
            n_average_samples=self.n_average_samples,  # Volumetric: average over N samples
            sharpening=self.sharpening,
            frequency_weighting=frequency_weighting,
            freq_cutoff_in_hz=self.filter_highcut  # Sync with filter setting
        )
        
        # Create tracker with current smoothing alpha
        tracker = ExponentialSmoothingTracker(alpha=self.smoothing_alpha)
        
        # Create streaming processor
        self.streaming_srp = StreamingSrp(
            srp_processor=srp_processor,
            tracker=tracker,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
            window_func=np.hanning,
            fs=self.fs,
            filter_enabled=self.filter_enabled,
            filter_lowcut=self.filter_lowcut,
            filter_highcut=self.filter_highcut
        )
        
        # Initialize polar plot
        self.polar_plot.initialize_plot(self.grid)
        
        # Reset radial max when reinitializing
        self.radial_max = None
        
        # Recompute noise floor map with new SRP settings if ambient audio is loaded
        # Only if not suppressed (suppressed when recomputation will be triggered separately)
        if not suppress_noise_floor_recompute and self.ambient_audio is not None:
            self.recompute_noise_floor(show_progress=False, reason="SRP processor initialized")
        
        return True
    
    def toggle_recording(self):
        """Start or stop recording."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start audio recording and processing."""
        if not self.validate_config():
            return
        
        # Show initialization progress
        init_progress = QProgressDialog("Initializing SRP processor...", None, 0, 0, self)
        init_progress.setWindowTitle("Starting Recording")
        init_progress.setWindowModality(Qt.WindowModal)
        init_progress.setMinimumDuration(0)
        init_progress.setValue(0)
        QApplication.processEvents()
        
        try:
            init_progress.setLabelText("Creating SRP processor...")
            QApplication.processEvents()
            
            if not self.initialize_srp_processor(suppress_noise_floor_recompute=True):
                init_progress.close()
                QMessageBox.warning(self, "Error", "Failed to initialize SRP processor")
                return
            
            # Only recompute noise floor if it hasn't been computed yet
            # (e.g., if ambient audio was loaded but SRP processor wasn't initialized)
            # If it was already computed (e.g., after calibration), skip recomputation
            if self.should_recompute_noise_floor() and self.noise_floor is None:
                # Calculate frames for progress
                n_samples, n_channels = self.ambient_audio.shape
                n_frames = (n_samples - self.frame_size) // self.hop_size + 1
                init_progress.setMaximum(n_frames)
                init_progress.setValue(0)
                
                # Create a progress callback that updates our dialog
                def noise_floor_progress(current, total):
                    progress_pct = int(100 * current / total) if total > 0 else 0
                    init_progress.setValue(current)
                    init_progress.setLabelText(
                        f"Computing noise floor map...\n"
                        f"Processing frame {current}/{total} ({progress_pct}%)"
                    )
                    QApplication.processEvents()
                    return not init_progress.wasCanceled()
                
                # Compute noise floor directly with our progress callback
                try:
                    cancelled = [False]
                    self.noise_floor = compute_noise_floor_map(
                        audio_data=self.ambient_audio,
                        srp_processor=self.streaming_srp.srp_processor,
                        frame_size=self.frame_size,
                        hop_size=self.hop_size,
                        fs=self.fs,
                        filter_enabled=self.filter_enabled,
                        filter_lowcut=self.filter_lowcut,
                        filter_highcut=self.filter_highcut,
                        ignore_channels=self.config.get('ignore_channels', []) if self.config else [],
                        progress_callback=noise_floor_progress
                    )
                    if init_progress.wasCanceled():
                        self.noise_floor = None
                        self.noise_floor_status_label.setText("Cancelled")
                    else:
                        self.noise_floor_status_label.setText("Loaded & Computed")
                except Exception as e:
                    print(f"Error computing noise floor during start: {e}")
                    self.noise_floor = None
            
            init_progress.setLabelText("Opening audio stream...")
            QApplication.processEvents()
            
            device_index = self.device_combo.currentData()
            device_info = self.audio.get_device_info_by_index(device_index)
            self.device_channels = device_info['maxInputChannels']
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.device_channels,
                rate=self.fs,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.hop_size,
                stream_callback=None
            )
            
            init_progress.setLabelText("Starting audio stream...")
            QApplication.processEvents()
            
            self.stream.start_stream()
            self.is_recording = True
            self.start_stop_button.setText("Stop Recording")
            self.status_label.setText("Status: Recording...")
            
            # Start processing timer (process every frame)
            self.timer.start(int(1000 * self.hop_size / self.fs))  # Update rate in ms
            
            init_progress.close()
            
        except Exception as e:
            init_progress.close()
            QMessageBox.critical(self, "Error", f"Failed to start recording: {e}")
            self.stop_recording()
    
    def stop_recording(self):
        """Stop audio recording and processing."""
        self.is_recording = False
        self.timer.stop()
        
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        self.start_stop_button.setText("Start Recording")
        self.status_label.setText("Status: Stopped")
    
    def process_audio_frame(self):
        """Process a single audio frame."""
        if not self.is_recording or self.stream is None:
            return
        
        try:
            # Read audio data
            audio_data = self.stream.read(self.hop_size, exception_on_overflow=False)
            
            # Convert to numpy array
            new_samples = np.frombuffer(audio_data, dtype=np.int16)
            new_samples = new_samples.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Reshape to (hop_size, channels)
            new_samples = new_samples.reshape(self.hop_size, self.device_channels)

            # Transpose to (channels, samples) for processing
            audio_chunk = new_samples.T
            
            # Filter out ignored channels
            ignore_channels = self.config.get('ignore_channels', [])
            valid_channels = [i for i in range(self.device_channels) if i not in ignore_channels]
            audio_chunk = audio_chunk[valid_channels, :]
            
            # Process chunk using internal buffer of StreamingSrp (filtering handled internally)
            srp_map, tracked_doa = self.streaming_srp.process_chunk(audio_chunk)
            
            # If tracking/processing returned None (e.g. not enough data), skip update
            if srp_map is None:
                return

            # Subtract noise floor if available and enabled
            if self.noise_floor_enabled and self.noise_floor is not None:
                # Ensure shapes match before subtraction
                # If shapes don't match, clear noise floor (likely due to parameter change)
                if srp_map.shape == self.noise_floor.shape:
                    srp_map = srp_map - self.noise_floor
                    # Ensure non-negative values
                    srp_map = np.maximum(srp_map, 0)
                else:
                    # Shape mismatch - clear noise floor to prevent errors
                    # This can happen if parameters changed but recomputation hasn't finished yet
                    # The noise floor will be recomputed by the parameter change handler
                    self.noise_floor = None
            
            # Update radial max if needed (use max of current frame * multiplier, or initialize)
            current_max = np.max(srp_map) * RADIAL_MAX_MULTIPLIER if np.max(srp_map) > 0 else 1.0
            if self.radial_max is None:
                self.radial_max = current_max
            else:
                # Gradually adapt to higher values, but don't shrink too quickly
                if current_max > self.radial_max:
                    self.radial_max = current_max
            
            # Update plot with fixed radial limits
            show_tracked = self.show_tracked_checkbox.isChecked()
            self.polar_plot.update_plot(srp_map, tracked_doa, radial_max=self.radial_max, 
                                       show_tracked=show_tracked)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_recording()
        if self.audio is not None:
            self.audio.terminate()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = SRPGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

