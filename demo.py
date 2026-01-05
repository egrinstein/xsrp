import sys
import os
import yaml
import numpy as np
import pyaudio
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QSlider, QCheckBox
)
from PyQt5.QtCore import QTimer, Qt, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from xsrp.conventional_srp import ConventionalSrp
from xsrp.streaming import StreamingSrp
from xsrp.tracking import ExponentialSmoothingTracker
from xsrp.grids import UniformSphericalGrid
from visualization.polar import plot_polar_srp_map


class PolarPlotWidget(QWidget):
    """Widget for displaying polar SRP map."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 8))
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


class SRPDemo(QMainWindow):
    """Main GUI application for real-time SRP demo."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time SRP Demo")
        self.setGeometry(100, 100, 1000, 800)
        
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
        
        # Processing parameters
        self.fs = 16000
        self.frame_size = 1024
        self.buffer_size = 2048
        self.smoothing_alpha = 0.7  # Default smoothing factor
        self.n_average_samples = 5  # Default averaging samples
        
        # Timer for real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_audio_frame)
        
        # Initialize UI
        self.init_ui()
        
        # Load default config
        default_config_path = Path(__file__).parent / "docs" / "mic_config" / "respeaker.yaml"
        if default_config_path.exists():
            self.load_config(str(default_config_path))
    
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
        self.smoothing_slider.setMinimum(0)
        self.smoothing_slider.setMaximum(100)
        self.smoothing_slider.setValue(70)  # Default 0.7
        self.smoothing_slider.valueChanged.connect(self.on_smoothing_changed)
        self.smoothing_value_label = QLabel("0.70")
        smoothing_layout.addWidget(self.smoothing_slider)
        smoothing_layout.addWidget(self.smoothing_value_label)
        controls_layout.addWidget(smoothing_label)
        controls_layout.addLayout(smoothing_layout)
        
        # Averaging control
        averaging_label = QLabel("Averaging Samples:")
        averaging_layout = QHBoxLayout()
        self.averaging_slider = QSlider(Qt.Horizontal)
        self.averaging_slider.setMinimum(1)
        self.averaging_slider.setMaximum(20)
        self.averaging_slider.setValue(5)  # Default 5
        self.averaging_slider.valueChanged.connect(self.on_averaging_changed)
        self.averaging_value_label = QLabel("5")
        averaging_layout.addWidget(self.averaging_slider)
        averaging_layout.addWidget(self.averaging_value_label)
        controls_layout.addWidget(averaging_label)
        controls_layout.addLayout(averaging_layout)
        
        # Show tracked position toggle
        self.show_tracked_checkbox = QCheckBox("Show Tracked Position")
        self.show_tracked_checkbox.setChecked(True)  # Default to showing
        self.show_tracked_checkbox.stateChanged.connect(self.on_show_tracked_changed)
        controls_layout.addWidget(self.show_tracked_checkbox)
        
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
        """Handle averaging slider change."""
        self.n_average_samples = value
        self.averaging_value_label.setText(str(value))
        # Reinitialize SRP processor if recording (since n_average_samples is set at initialization)
        if self.is_recording and self.streaming_srp is not None:
            if not self.initialize_srp_processor():
                QMessageBox.warning(self, "Error", "Failed to reinitialize SRP processor")
    
    def on_show_tracked_changed(self, state):
        """Handle show tracked position checkbox change."""
        # The next frame update will use the new setting
        pass
    
    def load_config_dialog(self):
        """Open dialog to load microphone configuration file."""
        default_path = Path(__file__).parent / "docs" / "mic_config"
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
            self.status_label.setText(f"Status: Config loaded ({len(mic_angles)} microphones)")
            
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
    
    def initialize_srp_processor(self):
        """Initialize the SRP processor with current configuration."""
        if self.mic_positions is None:
            return False
        
        # Create DOA grid (azimuth only)
        n_azimuth_cells = 72  # 5 degree resolution
        self.grid = UniformSphericalGrid(n_azimuth_cells)
        
        # For 1D DOA grid, use 2D mic positions (x, y only)
        # The grid positions are 2D, so mic positions must also be 2D
        mic_positions_2d = self.mic_positions[:, :2]  # Take only x, y coordinates
        
        # Create SRP processor with volumetric approach (n_average_samples > 1)
        # Volumetric SRP averages over N-closest correlation values for better robustness
        srp_processor = ConventionalSrp(
            fs=self.fs,
            grid_type="doa_1D",
            n_grid_cells=n_azimuth_cells,
            mic_positions=mic_positions_2d,
            mode="gcc_phat_time",
            interpolation=True,
            n_average_samples=self.n_average_samples  # Volumetric: average over N samples
        )
        
        # Create tracker with current smoothing alpha
        tracker = ExponentialSmoothingTracker(alpha=self.smoothing_alpha)
        
        # Create streaming processor
        self.streaming_srp = StreamingSrp(
            srp_processor=srp_processor,
            tracker=tracker,
            frame_size=self.frame_size
        )
        
        # Initialize polar plot
        self.polar_plot.initialize_plot(self.grid)
        
        # Reset radial max when reinitializing
        self.radial_max = None
        
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
        
        if not self.initialize_srp_processor():
            QMessageBox.warning(self, "Error", "Failed to initialize SRP processor")
            return
        
        device_index = self.device_combo.currentData()
        device_info = self.audio.get_device_info_by_index(device_index)
        self.device_channels = device_info['maxInputChannels']
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.device_channels,
                rate=self.fs,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.frame_size,
                stream_callback=None
            )
            
            self.stream.start_stream()
            self.is_recording = True
            self.start_stop_button.setText("Stop Recording")
            self.status_label.setText("Status: Recording...")
            
            # Start processing timer (process every frame)
            self.timer.start(int(1000 * self.frame_size / self.fs))  # Update rate in ms
            
        except Exception as e:
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
            audio_data = self.stream.read(self.frame_size, exception_on_overflow=False)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Reshape to (channels, samples)
            audio_array = audio_array.reshape(self.device_channels, self.frame_size)
            
            # Filter out ignored channels
            ignore_channels = self.config.get('ignore_channels', [])
            valid_channels = [i for i in range(self.device_channels) if i not in ignore_channels]
            audio_frame = audio_array[valid_channels, :]
            
            # Process frame
            srp_map, tracked_doa = self.streaming_srp.process_frame(audio_frame)
            
            # Update radial max if needed (use max of current frame * 1.1, or initialize)
            current_max = np.max(srp_map) * 1.1 if np.max(srp_map) > 0 else 1.0
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
    window = SRPDemo()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

