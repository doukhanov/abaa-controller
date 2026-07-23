import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import time
import random
import shutil
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QLineEdit, QComboBox, QFileDialog, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton, QMessageBox,
                            QTabWidget, QSplitter, QFrame, QListWidget, QProgressBar,
                            QDialog, QGridLayout, QTreeWidget, QTreeWidgetItem, QHeaderView,
                            QCheckBox, QButtonGroup, QScrollArea, QTabWidget, QTableWidget,
                            QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QDir
from PyQt5.QtGui import QFont, QIcon, QPen, QColor
import pyqtgraph as pg

# Import the model training functions from the second file
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib
import glob
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Missing import
from preprocess_single_csv import PreprocessingThread

# Add pyvisa for hardware communication
import pyvisa
import re  # Add this import at the top with other imports

# Set dark theme for pyqtgraph
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# Define larger base font size
FONT_SIZE_MULTIPLIER = 2.5
BASE_FONT_SIZE = int(10 * FONT_SIZE_MULTIPLIER)  # Base size (default is 10)
HEADER_FONT_SIZE = int(18 * FONT_SIZE_MULTIPLIER)  # Size for headers
BUTTON_FONT_SIZE = int(12 * FONT_SIZE_MULTIPLIER)  # Size for buttons
LABEL_FONT_SIZE = int(10 * FONT_SIZE_MULTIPLIER)  # Size for labels


# Directory structure constants
# Override with ABAA_DATA_DIR when running on another machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = os.environ.get("ABAA_DATA_DIR", str(PROJECT_ROOT / "runtime_data"))
USERS_DIR = os.path.join(BASE_DIR, "users")
DATA_DIR = USERS_DIR
TRAINED_DIR = "trained"  # Subdirectory for files the model was trained on (in each wafer dir)
UNTRAINED_DIR = "untrained"  # Subdirectory for files the model was not trained on (in each wafer dir)
MODEL_DIR = "models"  # Subdirectory for model files (in each wafer dir)
COOLED_RAW_DIR = "cooled_raw"  # Subdirectory for raw cooled measurement files
DEAD_DIR = "dead"  # Subdirectory for dead chips and invalid timing measurements

# Hardware constants
KEITHLEY_RESOURCE_NAME = os.environ.get(
    'ABAA_KEITHLEY_RESOURCE',
    'USB0::0x05E6::0x2450::<serial>::INSTR',
)  # Keithley 2450 resource identifier
MEASUREMENT_VOLTAGE = 0.01  # 10 mV for safe resistance measurements
# Add this constant at the top of your file with other constants
HARDWARE_CONNECTED = True  # Global flag to check hardware availability

def check_hardware_connection(self):
    """Check if the hardware is connected and set global flag"""
    global HARDWARE_CONNECTED
    try:
        rm = pyvisa.ResourceManager()
        keithley = rm.open_resource(KEITHLEY_RESOURCE_NAME)
        keithley.write("*IDN?")
        idn = keithley.read()
        keithley.close()
        
        # Display hardware info in status bar
        self.statusBar().showMessage(f"Connected to: {idn.strip()}")
        HARDWARE_CONNECTED = True
        return True
        
    except Exception as e:
        # Hardware not found
        self.statusBar().showMessage(f"Hardware not found. Running in demo mode.")
        print(f"Hardware connection error: {str(e)}")
        HARDWARE_CONNECTED = False
        return False

class ConservativeRegressor(BaseEstimator, RegressorMixin):
    """Wrapper that makes any regressor's predictions more conservative"""
    
    def __init__(self, base_estimator, scale_factor=1.1):
        self.base_estimator = base_estimator
        self.scale_factor = scale_factor
        
    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        return self
        
    def predict(self, X):
        predictions = self.base_estimator_.predict(X)
        # Apply scaling factor to intentionally underestimate
        return predictions * self.scale_factor
    
    def get_prediction_std(self, X):
        """Get standard deviation of predictions from forest estimator"""
        if hasattr(self.base_estimator_, "estimators_"):
            # For Random Forest, get predictions from all trees
            predictions = np.array([tree.predict(X) for tree in self.base_estimator_.estimators_])
            # Calculate standard deviation across trees
            return np.std(predictions, axis=0)
        else:
            # Default value if not a forest estimator
            return np.ones(X.shape[0]) * 100.0  # Default uncertainty of 100 ohms
    
    def get_params(self, deep=True):
        return {
            'base_estimator': self.base_estimator,
            'scale_factor': self.scale_factor
        }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

class MeasurementThread(QThread):
    """Thread for background measurements to keep UI responsive"""
    measurement_update = pyqtSignal(float, float)  # time, resistance
    measurement_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initial_measurement_thread = None
        self.initial_r_values = []
        self.is_measuring = False
        self.initial_r_avg = None
        self.initial_resistance = None
        self.running = False
        self.start_time = 0
        self.resistance_data = []
        self.time_data = []
        self.voltage_data = []
        self.keithley = None
        
    def run(self):
        """Main measurement loop"""
        global HARDWARE_CONNECTED
        if not HARDWARE_CONNECTED:
            # Use simulated measurements
            self.simulate_measurements()
            return
        self.running = True
        self.start_time = time.time()
        
        try:
            # Initialize hardware connection
            try:
                rm = pyvisa.ResourceManager()
                self.keithley = rm.open_resource(KEITHLEY_RESOURCE_NAME)
                # Initialize the instrument for resistance measurements
                self.keithley.write("*RST")
                self.keithley.write(":SOUR:FUNC VOLT")
                self.keithley.write(":SOUR:VOLT:RANG 1")
                self.keithley.write(":SENS:FUNC 'CURR'")
                self.keithley.write(":SENS:CURR:RANG 1E-3")
                print("Hardware connection established")
            except Exception as e:
                self.error_occurred.emit(f"Error connecting to hardware: {str(e)}")
                self.running = False
                return
            
            while self.running:
                # Get current time
                current_time = time.time() - self.start_time
                
                # Perform actual measurement
                try:
                    # Set measurement voltage
                    self.keithley.write(f":SOUR:VOLT {MEASUREMENT_VOLTAGE}")
                    self.keithley.write(":OUTP ON")
                    
                    # Measure current
                    current = float(self.keithley.query(":READ?"))
                    
                    # Calculate resistance (Ohm's law: R = V/I)
                    resistance = MEASUREMENT_VOLTAGE / current if abs(current) > 1e-12 else float('inf')
                    resistance = abs(resistance)  # Ensure positive value
                    voltage = MEASUREMENT_VOLTAGE
                    
                    # Turn off output after measurement
                    self.keithley.write(":OUTP OFF")
                except Exception as e:
                    self.error_occurred.emit(f"Error during hardware measurement: {str(e)}")
                    break
                
                # Store data
                self.time_data.append(current_time)
                self.resistance_data.append(resistance)
                self.voltage_data.append(voltage)
                
                # Emit signal with the new measurement
                self.measurement_update.emit(current_time, resistance)
                
                # Sleep to control measurement rate
                self.msleep(1000)  # 1 second between measurements
                
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Close hardware connection if it was opened
            if self.keithley is not None:
                try:
                    self.keithley.write(":OUTP OFF")
                    self.keithley.close()
                    print("Hardware connection closed")
                except:
                    pass
                    
            self.running = False
            self.measurement_complete.emit()
    
    def stop(self):
        """Stop the measurement thread"""
        self.running = False
        
        # Ensure hardware is turned off when stopping
        if self.keithley is not None:
            try:
                self.keithley.write(":OUTP OFF")
            except:
                pass
        
    def get_data(self):
        """Return the collected data"""
        return self.time_data, self.resistance_data, self.voltage_data
    
class BakingThread(QThread):
    """Thread for baking process with alternating bias"""
    measurement_update = pyqtSignal(float, float, float)  # time, resistance, voltage
    prediction_update = pyqtSignal(float)  # predicted_resistance
    measurement_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None, model_path=None, voltage=0.8, frequency=1.0, current_limit=800e-6):
        super().__init__(parent)
        self.running = False
        self.start_time = 0
        self.resistance_data = []
        self.time_data = []
        self.voltage_data = []
        self.model_path = model_path
        self.model = None
        self.target_resistance = 0
        self.auto_stop = False
        self.offset_ohms = 0
        self.keithley = None
        self.predicted_final_r = None
        self.baking_end_timestamp = None
        self.final_baking_resistance = None
        self.instrument_connected = False  # Add this flag
        self.temperature = 100.0  # Default, will be updated from UI
        self.initial_resistance = None  # Will be set when baking starts
        
        # Store the actual UI parameters
        self.voltage = voltage
        self.frequency = frequency
        self.current_limit = current_limit
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                
    def compute_real_time_features(self, current_r, valid_resistances, valid_times, pulse_count):
        """
        Compute features using only the first 10 data points for early prediction.
        Returns percentage change prediction.
        """
        try:
            # Wait until we have at least 10 data points
            if len(valid_resistances) < 10:
                return None
            
            # Get parameters set before measurement began
            temperature = getattr(self, 'temperature', 100.0)
            initial_resistance = getattr(self, 'initial_resistance', valid_resistances[0] if valid_resistances else current_r)
            voltage = getattr(self, 'voltage', 0.8)
            frequency = getattr(self, 'frequency', 1.0)
            
            # Use only first 10 data points
            first_10_times = np.array(valid_times[:10])
            first_10_resistances = np.array(valid_resistances[:10])
            
            # Pre-baking features
            first_baking = first_10_resistances[0]
            jump_absolute = first_baking - initial_resistance
            jump_pct = (jump_absolute / initial_resistance) * 100 if initial_resistance > 0 else 0
            initial_drop_pct = jump_pct  # CRITICAL FEATURE - this is the initial drop percentage
            
            # Baking features from first 10 points ONLY
            baking_time_10pt = first_10_times[-1] - first_10_times[0] if len(first_10_times) > 1 else 0
            last_baking_10pt = first_10_resistances[-1]  # 10th point, not current
            
            # Calculate derived features using only first 10 points
            baking_change_10pt = last_baking_10pt - first_baking
            baking_pct_change_10pt = (baking_change_10pt / first_baking) * 100 if first_baking > 0 else 0
            
            # First derivative (baking slope) from first 10 points
            baking_slope_10pt = 0
            if len(first_10_times) >= 2:
                try:
                    from scipy import stats
                    slope, _, _, _, _ = stats.linregress(first_10_times, first_10_resistances)
                    baking_slope_10pt = slope
                except:
                    baking_slope_10pt = (first_10_resistances[-1] - first_10_resistances[0]) / (first_10_times[-1] - first_10_times[0])
            
            # Mean rate from first 10 points
            mean_rate_10pt = 0
            if len(first_10_resistances) >= 2:
                time_diffs = np.diff(first_10_times)
                resistance_diffs = np.diff(first_10_resistances)
                rates = resistance_diffs / np.maximum(time_diffs, 0.001)
                mean_rate_10pt = np.mean(rates)
            
            # Second derivative from first 10 points
            second_derivative_10pt = 0
            if len(first_10_resistances) >= 3:
                try:
                    # Calculate first derivatives
                    time_diffs = np.diff(first_10_times)
                    resistance_diffs = np.diff(first_10_resistances)
                    first_derivatives = resistance_diffs / np.maximum(time_diffs, 0.001)
                    
                    # Calculate second derivatives
                    if len(first_derivatives) >= 2:
                        midpoint_times = (first_10_times[1:] + first_10_times[:-1]) / 2
                        time_diffs_2nd = np.diff(midpoint_times)
                        second_derivative_values = np.diff(first_derivatives) / np.maximum(time_diffs_2nd, 0.001)
                        second_derivative_10pt = np.mean(second_derivative_values)
                except Exception as e:
                    print(f"Error calculating second derivative: {e}")
                    second_derivative_10pt = 0
            
            # Create weighted jump features (giving MAXIMUM weight to initial drop)
            initial_drop_pct_5x = initial_drop_pct * 5.0   # 5x weight - CRITICAL
            initial_drop_pct_10x = initial_drop_pct * 10.0  # 10x weight - CRITICAL
            jump_absolute_weighted_3x = jump_absolute * 3.0  # 3x weight
            
            print(f"CRITICAL FEATURE - initial_drop_pct: {initial_drop_pct:.2f}% (5x: {initial_drop_pct_5x:.2f}, 10x: {initial_drop_pct_10x:.2f})")
            print(f"10-point features: time={baking_time_10pt:.1f}s, change={baking_change_10pt:.2f}Ω ({baking_pct_change_10pt:.1f}%), slope={baking_slope_10pt:.3f}, rate={mean_rate_10pt:.3f}, accel={second_derivative_10pt:.3f}")
            
            return {
                # Pre-baking features - CRITICAL FEATURE FIRST
                "initial_drop_pct": initial_drop_pct,  # MOST CRITICAL FEATURE
                "initial_drop_pct_5x": initial_drop_pct_5x,   # CRITICAL with 5x weight
                "initial_drop_pct_10x": initial_drop_pct_10x,  # CRITICAL with 10x weight
                "temperature": temperature,
                "voltage": voltage,
                "frequency": frequency,
                "initial_resistance": initial_resistance,
                "first_baking_resistance": first_baking,
                "jump_absolute": jump_absolute,
                "jump_pct": jump_pct,
                "jump_absolute_weighted_3x": jump_absolute_weighted_3x,
                
                # Baking features from first 10 points ONLY
                "baking_time": baking_time_10pt,
                "last_baking_resistance": last_baking_10pt,
                "baking_change": baking_change_10pt,
                "baking_pct_change": baking_pct_change_10pt,
                "baking_slope": baking_slope_10pt,
                "mean_rate": mean_rate_10pt,
                "second_derivative": second_derivative_10pt
            }
            
        except Exception as e:
            print(f"Error computing real-time features: {e}")
            return None

    def set_target(self, resistance, auto_stop=False, offset_ohms=0):
        """Set target resistance and auto-stop parameters"""
        self.target_resistance = resistance
        self.auto_stop = auto_stop
        self.offset_ohms = offset_ohms
        
    def run(self):
        """Main baking loop with alternating bias using the exact voltage profile from Jupyter"""
        self.running = True
        self.start_time = time.time()
        
        # Initialize data collections
        self.resistance_data = []
        self.time_data = []
        self.voltage_data = []
        
        # Use the actual UI parameters
        V = self.voltage      # Voltage amplitude from UI
        freq = self.frequency # Frequency from UI
        current_limit = self.current_limit  # Current limit from UI
        
        print(f"Starting baking with: Voltage={V}V, Frequency={freq}Hz, Current Limit={current_limit*1e6}µA")
        
        # Timing parameters - EXACT same as Jupyter notebook
        T_high = 1 / freq         # Duration of high-voltage pulse (+V or -V)
        T_low = T_high / 10       # Duration of low-voltage measurement period
        n_high_steps = 5          # Number of data points during high voltage
        n_low_steps = 3           # Number of data points during low voltage
        dt_high = T_high / n_high_steps       
        dt_low = T_low / n_low_steps
        low_voltage = 0.01        # Safe low voltage for resistance measurement (same as MEASUREMENT_VOLTAGE)
        
        pulse_count = 0
        
        try:
            # Initialize hardware connection
            try:
                rm = pyvisa.ResourceManager()
                self.keithley = rm.open_resource(KEITHLEY_RESOURCE_NAME)
                # Initialize the instrument for voltage source
                self.keithley.write("*RST")
                self.keithley.write(":SOUR:FUNC VOLT")
                self.keithley.write(":SOUR:VOLT:RANG 1")
                self.keithley.write(":SENS:FUNC 'CURR'")
                self.keithley.write(":SENS:CURR:RANG 1E-3")
                # Set the current limit from UI
                self.keithley.write(f":SOUR:VOLT:ILIM {current_limit}")
                self.instrument_connected = True  # Set flag on successful connection
                print(f"Hardware connection established for baking with {current_limit*1e6}µA current limit")
            except Exception as e:
                self.instrument_connected = False  # Set flag on failed connection
                print(f"Running in demo mode: {str(e)}")
                # Don't emit error - continue in demo mode
            
            # Turn on output once at the beginning (only if hardware connected)
            if self.instrument_connected:
                self.keithley.write(":OUTP ON")
            
            while self.running:
                pulse_count += 1
                
                # --- Positive Voltage Pulse (+V) ---
                if self.instrument_connected:
                    self.keithley.write(f":SOUR:VOLT {V}")
                
                # Record data points for the high voltage pulse
                for i in range(n_high_steps):
                    if not self.running:
                        break
                        
                    current_time = time.time() - self.start_time
                    
                    if self.instrument_connected:
                        try:
                            current = float(self.keithley.query(":READ?"))
                            resistance = float('nan')  # No meaningful resistance during high voltage
                        except Exception as e:
                            self.error_occurred.emit(f"Error reading current: {str(e)}")
                            resistance = float('nan')
                    else:
                        # Demo mode
                        resistance = float('nan')
                    
                    self.time_data.append(current_time)
                    self.voltage_data.append(V)
                    self.resistance_data.append(resistance)
                    
                    # Emit signal
                    self.measurement_update.emit(current_time, resistance, V)
                    
                    self.msleep(int(dt_high * 1000))  # Convert to milliseconds
                
                if not self.running:
                    break
                
                # --- Low Voltage Measurement after +V ---
                if self.instrument_connected:
                    self.keithley.write(f":SOUR:VOLT {low_voltage}")
                
                for i in range(n_low_steps):
                    if not self.running:
                        break
                        
                    current_time = time.time() - self.start_time
                    
                    if self.instrument_connected:
                        try:
                            current = float(self.keithley.query(":READ?"))
                            resistance = low_voltage / current if abs(current) > 1e-12 else float('inf')
                            resistance = abs(resistance)  # Ensure positive value
                        except Exception as e:
                            self.error_occurred.emit(f"Error measuring resistance: {str(e)}")
                            break
                    else:
                        # Demo mode - simulate resistance with drift
                        base = 2000 + pulse_count * 20
                        resistance = base + (-1)**pulse_count * 100
                    
                    # Only store measurement on the last low voltage step
                    if i == n_low_steps - 1:
                        self.time_data.append(current_time)
                        self.voltage_data.append(low_voltage)
                        self.resistance_data.append(resistance)
                        
                        # Emit signal
                        self.measurement_update.emit(current_time, resistance, low_voltage)
                        
                        # *** PREDICTION SECTION - CORRECTED CONSTRAINT LOGIC ***
                        if self.model and len(self.resistance_data) >= 10:
                            # Get valid data (not NaN)
                            valid_indices = [i for i, r in enumerate(self.resistance_data) if not np.isnan(r)]
                            
                            if len(valid_indices) >= 10:
                                try:
                                    valid_resistances = [self.resistance_data[i] for i in valid_indices]
                                    valid_times = [self.time_data[i] for i in valid_indices]
                                    current_r = valid_resistances[-1]
                                    
                                    # Compute features for prediction
                                    feature_dict = self.compute_real_time_features(current_r, valid_resistances, valid_times, pulse_count)
                                    
                                    if feature_dict is not None:
                                        # Get the initial drop percentage for bounds calculation
                                        initial_drop_pct = feature_dict.get('initial_drop_pct', 0)
                                        initial_resistance = feature_dict.get('initial_resistance', current_r)
                                        
                                        # Calculate bounds based on MAGNITUDE of initial drop
                                        # If initial drop was -10%, expected rise should be between +16% and +22%
                                        # Bounds are 1.5x to 2.3x the MAGNITUDE (absolute value) of the initial drop
                                        if initial_drop_pct != 0:
                                            drop_magnitude = abs(initial_drop_pct)  # Get magnitude (remove negative sign)
                                            min_pct_change = drop_magnitude * 1.5   # Minimum positive rise
                                            max_pct_change = drop_magnitude * 2.3   # Maximum positive rise
                                            
                                            print(f"Initial drop: {initial_drop_pct:.2f}% (magnitude: {drop_magnitude:.2f}%)")
                                            print(f"Expected rise bounds: [+{min_pct_change:.2f}%, +{max_pct_change:.2f}%]")
                                        else:
                                            # Fallback if no initial drop detected
                                            min_pct_change = 10   # Default minimum 10% rise
                                            max_pct_change = 50   # Default maximum 50% rise
                                            print(f"No initial drop detected, using default bounds: [+{min_pct_change:.2f}%, +{max_pct_change:.2f}%]")
                                        
                                        # Check if model has metadata about selected features
                                        if hasattr(self.model, 'metadata') and 'selected_features' in self.model.metadata:
                                            selected_features = self.model.metadata['selected_features']
                                        else:
                                            # Fallback: use all available features except target
                                            selected_features = [key for key in feature_dict.keys() if 'change' not in key.lower()]
                                        
                                        # Create feature array in the correct order
                                        feature_values = []
                                        for feature_name in selected_features:
                                            if feature_name in feature_dict:
                                                feature_values.append(feature_dict[feature_name])
                                            else:
                                                print(f"Warning: Feature {feature_name} not found, using 0")
                                                feature_values.append(0.0)
                                        
                                        # Create input array for prediction
                                        X_input = np.array([feature_values])
                                        
                                        # Make raw prediction using the trained model
                                        raw_predicted_pct_change = self.model.predict(X_input)[0]
                                        
                                        # *** APPLY CONSTRAINTS: Clip to positive range based on initial drop magnitude ***
                                        constrained_pct_change = np.clip(raw_predicted_pct_change, min_pct_change, max_pct_change)
                                        
                                        # Calculate final resistance prediction: current_resistance * (1 + pct_change/100)
                                        predicted_final_r = current_r * (1 + constrained_pct_change / 100)
                                        
                                        # Apply offset if needed
                                        predicted_final_r = predicted_final_r + self.offset_ohms
                                        
                                        self.predicted_final_r = predicted_final_r
                                        
                                        # Enhanced logging to show constraint application
                                        if abs(raw_predicted_pct_change - constrained_pct_change) > 0.1:
                                            print(f"Model prediction CONSTRAINED:")
                                            print(f"  Raw prediction: {raw_predicted_pct_change:.2f}%")
                                            print(f"  Constrained to: +{constrained_pct_change:.2f}% (bounds: [+{min_pct_change:.2f}%, +{max_pct_change:.2f}%])")
                                            print(f"  Final resistance: {predicted_final_r:.2f}Ω (Current={current_r:.2f}Ω)")
                                        else:
                                            print(f"Model prediction: Rise=+{constrained_pct_change:.2f}%, Final={predicted_final_r:.2f}Ω (Current={current_r:.2f}Ω)")
                                        
                                        # Emit prediction update
                                        self.prediction_update.emit(predicted_final_r)
                                        
                                        # Check for auto-stop condition
                                        if self.auto_stop and self.target_resistance > 0:
                                            if abs(predicted_final_r - self.target_resistance) <= 50:  # 50 ohms tolerance
                                                print("Auto-stop condition met!")
                                                self.running = False
                                                break
                                                
                                except Exception as e:
                                    print(f"Prediction error: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    
                                    # *** CORRECTED FALLBACK PREDICTION ***
                                    try:
                                        # Try to get initial drop from stored initial resistance
                                        if hasattr(self, 'initial_resistance') and self.initial_resistance:
                                            first_valid_r = valid_resistances[0] if valid_resistances else current_r
                                            fallback_initial_drop = ((first_valid_r - self.initial_resistance) / self.initial_resistance) * 100
                                            
                                            # Apply constraint logic: positive rise from negative drop magnitude
                                            if fallback_initial_drop != 0:
                                                drop_magnitude = abs(fallback_initial_drop)
                                                fallback_min = drop_magnitude * 1.5  # Minimum positive rise
                                                fallback_max = drop_magnitude * 2.3  # Maximum positive rise
                                                
                                                # Use middle of the range as fallback
                                                fallback_pct_change = (fallback_min + fallback_max) / 2
                                            else:
                                                fallback_pct_change = 30  # Default 30% positive rise
                                        else:
                                            fallback_pct_change = 30  # Default 30% positive rise
                                        
                                        predicted_final_r = current_r * (1 + fallback_pct_change / 100) + self.offset_ohms
                                        self.predicted_final_r = predicted_final_r
                                        self.prediction_update.emit(predicted_final_r)
                                        
                                        print(f"Fallback constrained prediction: +{fallback_pct_change:.2f}% → {predicted_final_r:.2f}Ω")
                                        
                                    except:
                                        # Last resort fallback
                                        predicted_final_r = current_r * 1.3 + self.offset_ohms  # Assume 30% rise
                                        self.predicted_final_r = predicted_final_r
                                        self.prediction_update.emit(predicted_final_r)
                                        print(f"Emergency fallback prediction: +30% → {predicted_final_r:.2f}Ω")
                    
                    self.msleep(int(dt_low * 1000))
                
                if not self.running:
                    break
                
                # --- Negative Voltage Pulse (-V) ---
                if self.instrument_connected:
                    self.keithley.write(f":SOUR:VOLT {-V}")
                
                # Record data points during negative high voltage
                for i in range(n_high_steps):
                    if not self.running:
                        break
                        
                    current_time = time.time() - self.start_time
                    
                    if self.instrument_connected:
                        try:
                            current = float(self.keithley.query(":READ?"))
                            resistance = float('nan')  # No meaningful resistance during high voltage
                        except Exception as e:
                            self.error_occurred.emit(f"Error reading current: {str(e)}")
                            resistance = float('nan')
                    else:
                        resistance = float('nan')
                    
                    self.time_data.append(current_time)
                    self.voltage_data.append(-V)  # Negative voltage
                    self.resistance_data.append(resistance)
                    
                    # Emit signal
                    self.measurement_update.emit(current_time, resistance, -V)
                    
                    self.msleep(int(dt_high * 1000))
                
                if not self.running:
                    break
                
                # --- Low Voltage Measurement after -V ---
                if self.instrument_connected:
                    self.keithley.write(f":SOUR:VOLT {low_voltage}")
                
                for i in range(n_low_steps):
                    if not self.running:
                        break
                        
                    current_time = time.time() - self.start_time
                    
                    if self.instrument_connected:
                        try:
                            current = float(self.keithley.query(":READ?"))
                            resistance = low_voltage / current if abs(current) > 1e-12 else float('inf')
                            resistance = abs(resistance)  # Ensure positive value
                        except Exception as e:
                            self.error_occurred.emit(f"Error measuring resistance: {str(e)}")
                            break
                    else:
                        # Demo mode
                        base = 2000 + pulse_count * 20
                        resistance = base + (-1)**pulse_count * 100
                    
                    # Only store measurement on the first low voltage step after -V
                    if i == 0:
                        self.time_data.append(current_time)
                        self.voltage_data.append(low_voltage)
                        self.resistance_data.append(resistance)
                        
                        # Emit signal
                        self.measurement_update.emit(current_time, resistance, low_voltage)
                    
                    self.msleep(int(dt_low * 1000))
        
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            # Store baking end timestamp and final resistance before stopping
            self.baking_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get the final baking resistance (last valid measurement)
            valid_resistances = [r for r in self.resistance_data if not np.isnan(r)]
            if valid_resistances:
                self.final_baking_resistance = valid_resistances[-1]
            
            # Turn off the instrument
            if self.keithley is not None and self.instrument_connected:
                try:
                    self.keithley.write(":SOUR:VOLT 0")
                    self.keithley.write(":OUTP OFF")
                    self.keithley.close()
                    print("Hardware connection closed for baking")
                except:
                    pass
                    
            self.running = False
            self.measurement_complete.emit()
    def stop(self):
        """Stop the baking process"""
        self.running = False
        
        # Ensure hardware is turned off when stopping
        if self.keithley is not None and self.instrument_connected:
            try:
                self.keithley.write(":SOUR:VOLT 0")
                self.keithley.write(":OUTP OFF")
            except:
                pass
        
    def get_data(self):
        """Return the collected data"""
        return self.time_data, self.resistance_data, self.voltage_data
    
class ModelTrainingThread(QThread):
    """Thread for training a new model with comprehensive feature extraction and hyperparameter tuning"""
    progress_update = pyqtSignal(int, str)  # percentage, status message
    training_complete = pyqtSignal(str)  # model path
    error_occurred = pyqtSignal(str)
    
    def __init__(self, wafer_dir, wafer_name, parent=None):
        super().__init__(parent)
        self.wafer_dir = wafer_dir
        self.wafer_name = wafer_name
        
    def run(self):
        """Run the model training process with feature extraction and hyperparameter tuning"""
        try:
            # Setup directories
            trained_dir = os.path.join(self.wafer_dir, TRAINED_DIR)
            untrained_dir = os.path.join(self.wafer_dir, UNTRAINED_DIR)
            model_dir = os.path.join(self.wafer_dir, MODEL_DIR)
            
            # Ensure model directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            # Generate model filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{self.wafer_name}_model_{timestamp}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            
            # Find all unified CSV files
            unified_files = []
            for directory in [trained_dir, untrained_dir]:
                if os.path.exists(directory):
                    files = glob.glob(os.path.join(directory, "*_unified_*.csv"))
                    unified_files.extend(files)
            
            if len(unified_files) < 5:
                self.error_occurred.emit("Not enough data files for training (minimum 5 required)")
                return
                
            # Report progress
            self.progress_update.emit(10, f"Loading training data from {len(unified_files)} files...")
            
            # Extract features and targets from all files
            all_chip_features = []
            
            for file_path in unified_files:
                try:
                    chip_features = self.extract_chip_features_from_unified(file_path)
                    if chip_features is not None:
                        all_chip_features.append(chip_features)
                except Exception as e:
                    print(f"⚠️ Error processing {file_path}: {str(e)}")
            
            if len(all_chip_features) < 5:
                self.error_occurred.emit("Not enough valid feature data for training")
                return
            
            # Convert to DataFrame
            df_features = pd.DataFrame(all_chip_features)
            
            # Fill NaN values with median
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_features[col].isna().any():
                    median_val = df_features[col].median()
                    df_features[col].fillna(median_val, inplace=True)
            
            self.progress_update.emit(30, f"Extracted features from {len(df_features)} chips")
            
            # Train the model
            self.progress_update.emit(50, "Training Random Forest model...")
            training_results = self.train_rf_model(df_features, target_col="cooled_pct_change")
            
            if training_results is None:
                self.error_occurred.emit("Model training failed")
                return
            
            # Save the model with comprehensive metadata
            self.progress_update.emit(80, "Saving the model...")
            
            # Get file counts
            untrained_files = [f for f in os.listdir(untrained_dir) if f.endswith('.csv')] if os.path.exists(untrained_dir) else []
            trained_files = [f for f in os.listdir(trained_dir) if f.endswith('.csv')] if os.path.exists(trained_dir) else []
            
            # Add comprehensive metadata to the model object
            training_results['model'].metadata = {
                'creation_date': timestamp,
                'wafer_name': self.wafer_name,
                'file_count': len(unified_files),
                'trained_files': len(trained_files),
                'untrained_files': len(untrained_files),
                'selected_features': training_results['selected_features'],
                'feature_importance': {
            'features': training_results['feature_importance']['Feature'].tolist(),
            'importance': training_results['feature_importance']['Importance'].tolist()},
                'metrics': training_results['metrics']
            }
            
            joblib.dump(training_results['model'], model_path)
            
            # Move untrained files to trained directory
            self.progress_update.emit(90, "Moving files to trained directory...")
            for f in untrained_files:
                src_path = os.path.join(untrained_dir, f)
                dst_path = os.path.join(trained_dir, f)
                if os.path.exists(src_path):  # Check if file still exists
                    shutil.move(src_path, dst_path)
            
            # Complete
            self.progress_update.emit(100, "Training complete!")
            self.training_complete.emit(model_path)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Training error: {str(e)}")

    # 1. FEATURE EXTRACTION FUNCTION (for training)
    def extract_chip_features_from_unified(self, file_path):
        """Extract features from a unified CSV file - TARGET: percentage change from last baking to cooled"""
        try:
            df = pd.read_csv(file_path)
            
            # Skip if not enough data
            if len(df) < 10:
                return None
            
            # Get summary data from first row
            cooled_resistance = df['Cooled_Average_Resistance'].iloc[0]
            initial_resistance = df['Initial_Resistance (Ohms)'].iloc[0]
            
            # Skip if missing critical data
            if pd.isna(cooled_resistance) or pd.isna(initial_resistance):
                return None
            
            # Extract baking time series data
            time_data = df['Time (s)'].values
            resistance_data = df['Resistance (Ohms)'].values
            voltage_data = df['Voltage (V)'].values
            
            # Remove NaN values
            valid_indices = ~np.isnan(resistance_data)
            time_data = time_data[valid_indices]
            resistance_data = resistance_data[valid_indices]
            voltage_data = voltage_data[valid_indices]
            
            if len(resistance_data) < 5:
                return None
            
            # Basic measurements
            first_baking = resistance_data[0]
            last_baking = resistance_data[-1]  # This is the "last baking resistance"
            baking_time = time_data[-1] - time_data[0] if len(time_data) > 1 else 0
            
            # Calculate derived features
            jump_absolute = first_baking - initial_resistance
            jump_pct = (jump_absolute / initial_resistance) * 100 if initial_resistance > 0 else 0
            initial_drop_pct = jump_pct  # CRITICAL FEATURE
            baking_change = last_baking - first_baking
            baking_pct_change = (baking_change / first_baking) * 100 if first_baking > 0 else 0
            
            # Create weighted initial drop features (CRITICAL - matching what training expects)
            initial_drop_pct_5x = initial_drop_pct * 5.0   # 5x weight - CRITICAL
            initial_drop_pct_10x = initial_drop_pct * 10.0  # 10x weight - CRITICAL
            jump_absolute_weighted_3x = jump_absolute * 3.0  # 3x weight
            
            # Extract voltage and frequency from filename or use defaults
            voltage = 0.8  # Default
            frequency = 1.0  # Default
            
            try:
                filename = os.path.basename(file_path)
                import re
                
                # Extract voltage (look for patterns like "0.8V" or "1.2V")
                voltage_match = re.search(r'(\d+(?:\.\d+)?)V', filename)
                if voltage_match:
                    voltage = float(voltage_match.group(1))
                
                # Extract frequency (look for patterns like "1.0Hz" or "2Hz")
                freq_match = re.search(r'(\d+(?:\.\d+)?)Hz', filename)
                if freq_match:
                    frequency = float(freq_match.group(1))
            except:
                pass
            
            # Temperature (extract from filename or use default)
            temperature = 100.0
            try:
                filename = os.path.basename(file_path)
                import re
                temp_match = re.search(r'(\d+(?:\.\d+)?)C', filename)
                if temp_match:
                    temperature = float(temp_match.group(1))
            except:
                pass
            
            # Calculate slope and rate features from first 10 points
            baking_slope = 0
            mean_rate = 0
            second_derivative = 0
            
            if len(resistance_data) >= 10:
                # Use first 10 points for early features
                first_10_times = time_data[:10]
                first_10_resistances = resistance_data[:10]
                
                # Calculate slope (first derivative)
                if len(first_10_times) >= 2:
                    try:
                        from scipy import stats
                        slope, _, _, _, _ = stats.linregress(first_10_times, first_10_resistances)
                        baking_slope = slope
                    except:
                        # Fallback simple slope calculation
                        baking_slope = (first_10_resistances[-1] - first_10_resistances[0]) / (first_10_times[-1] - first_10_times[0])
                
                # Calculate mean rate
                if len(first_10_resistances) >= 2:
                    time_diffs = np.diff(first_10_times)
                    resistance_diffs = np.diff(first_10_resistances)
                    rates = resistance_diffs / np.maximum(time_diffs, 0.001)
                    mean_rate = np.mean(rates)
                
                # Calculate second derivative (acceleration)
                if len(first_10_resistances) >= 3:
                    try:
                        # Calculate first derivatives
                        time_diffs = np.diff(first_10_times)
                        resistance_diffs = np.diff(first_10_resistances)
                        first_derivatives = resistance_diffs / np.maximum(time_diffs, 0.001)
                        
                        # Calculate second derivatives
                        if len(first_derivatives) >= 2:
                            midpoint_times = (first_10_times[1:] + first_10_times[:-1]) / 2
                            time_diffs_2nd = np.diff(midpoint_times)
                            second_derivative_values = np.diff(first_derivatives) / np.maximum(time_diffs_2nd, 0.001)
                            second_derivative = np.mean(second_derivative_values)
                    except Exception as e:
                        print(f"Error calculating second derivative: {e}")
                        second_derivative = 0
            
            # *** NEW TARGET: Calculate percentage change from last baking to cooled ***
            if last_baking > 0:
                cooled_pct_change = ((cooled_resistance - last_baking) / last_baking) * 100
            else:
                cooled_pct_change = 0
            
            print(f"File: {os.path.basename(file_path)}")
            print(f"  Last baking: {last_baking:.2f}Ω → Cooled: {cooled_resistance:.2f}Ω")
            print(f"  Percentage change: {cooled_pct_change:.2f}%")
            
            return {
                # CRITICAL FEATURES FIRST (matching training expectations)
                "initial_drop_pct": initial_drop_pct,  # MOST CRITICAL FEATURE
                "initial_drop_pct_5x": initial_drop_pct_5x,   # CRITICAL with 5x weight
                "initial_drop_pct_10x": initial_drop_pct_10x,  # CRITICAL with 10x weight
                
                # Pre-baking features
                "temperature": temperature,
                "voltage": voltage,
                "frequency": frequency,
                "initial_resistance": initial_resistance,
                "first_baking_resistance": first_baking,
                "jump_absolute": jump_absolute,
                "jump_pct": jump_pct,
                "jump_absolute_weighted_3x": jump_absolute_weighted_3x,
                
                # Baking features
                "baking_time": baking_time,
                "last_baking_resistance": last_baking,
                "baking_change": baking_change,
                "baking_pct_change": baking_pct_change,
                "baking_slope": baking_slope,
                "mean_rate": mean_rate,
                "second_derivative": second_derivative,
                
                # *** NEW TARGET VARIABLE: percentage change from last baking to cooled ***
                "cooled_pct_change": cooled_pct_change,  # TARGET VARIABLE
            }
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return None

        
    def optimize_feature_subset(self, df_features, target_col="resistance_rise", random_state=42):
        """Optimize feature subset using k-fold cross validation for 7-9 features"""
        from sklearn.model_selection import train_test_split, KFold, GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import pandas as pd
        import numpy as np
        
        # All numeric columns except target columns
        target_columns = [col for col in df_features.columns if 'rise' in col.lower()]
        all_features = [col for col in df_features.columns if col not in target_columns]
        
        # Preliminary RF to rank features
        prelim_rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        X_all = df_features[all_features]
        y_all = df_features[target_col]
        prelim_rf.fit(X_all, y_all)
        
        importances = prelim_rf.feature_importances_
        feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
        ranked_features = feat_imp['Feature'].tolist()
        
        print("Preliminary feature ranking:")
        print(feat_imp.head(15))
        
        best_score = np.inf
        best_k = None
        
        # Try different numbers of features (7-9 as requested)
        for k in range(5, 10 ,2):
            if k > len(ranked_features):
                break
                
            selected = ranked_features[:k]
            X = df_features[selected]
            y = df_features[target_col]
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=random_state)
            
            # Quick pipeline for validation
            rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1, min_samples_leaf=1)
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rf', rf_base)
            ])
            param_grid = {
                'rf__n_estimators': [100],
                'rf__max_depth': [10],
            }
            kfold = KFold(n_splits=3, shuffle=True, random_state=random_state)
            gs = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
            gs.fit(X_train, y_train)
            
            rmse = np.sqrt(-gs.best_score_)
            print(f"k={k} => CV RMSE {rmse:.2f}")
            if rmse < best_score:
                best_score = rmse
                best_k = k
        
        final_features = ranked_features[:best_k]
        print(f"Best number of features: {best_k}, using {final_features}")
        return final_features

        # Updated training function to analyze positive rise from negative drops
    def train_rf_model(self, df_features, target_col="cooled_pct_change", test_size=0.2, random_state=None):
        """Train model to predict POSITIVE percentage rise with lightweight regularization"""
        from sklearn.model_selection import train_test_split, KFold, GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import pandas as pd
        import numpy as np
        
        # Use current time as random seed if none provided
        if random_state is None:
            import time
            random_state = int(time.time()) % 10000
        
        print(f"Using random_state: {random_state}")
        print(f"\n🧠 Training REGULARIZED model to predict POSITIVE rise (1.5x to 2.3x magnitude of initial drop)...")
        
        # Use ALL features except target columns
        target_columns = [col for col in df_features.columns if 'change' in col.lower() and col != target_col]
        all_features = [col for col in df_features.columns if col not in target_columns + [target_col]]
        
        # ENSURE critical features are prioritized but check if they exist
        critical_features = ['initial_drop_pct', 'initial_drop_pct_5x', 'initial_drop_pct_10x']
        available_critical = [f for f in critical_features if f in all_features]
        other_features = [f for f in all_features if f not in critical_features]
        
        # Order features with available critical features first
        ordered_features = available_critical + other_features
        
        print(f"🔥 CRITICAL FEATURES (available): {available_critical}")
        print(f"Using {len(ordered_features)} total features with critical features prioritized")
        print(f"🎯 TARGET: {target_col} (POSITIVE percentage rise from last baking to cooled)")
        
        X_all = df_features[ordered_features]
        y_all = df_features[target_col]
        
        # *** ANALYZE CONSTRAINT ADHERENCE IN TRAINING DATA ***
        if 'initial_drop_pct' in df_features.columns:
            initial_drops = df_features['initial_drop_pct']
            
            # Calculate expected bounds for each sample (positive rise from negative drop magnitude)
            constraint_violations = 0
            total_samples = len(df_features)
            
            print(f"\n📊 Training Data Constraint Analysis:")
            print(f"{'Sample':<8} {'Initial Drop':<12} {'Expected Range':<20} {'Actual Rise':<12} {'Valid?':<6}")
            print("-" * 70)
            
            for idx, (initial_drop, actual_change) in enumerate(zip(initial_drops, y_all)):
                if initial_drop != 0:
                    drop_magnitude = abs(initial_drop)  # Get magnitude
                    expected_min = drop_magnitude * 1.5  # Minimum positive rise
                    expected_max = drop_magnitude * 2.3  # Maximum positive rise
                    
                    is_valid = expected_min <= actual_change <= expected_max
                    if not is_valid:
                        constraint_violations += 1
                    
                    # Show first 10 samples for analysis
                    if idx < 10:
                        print(f"{idx+1:<8} {initial_drop:<12.2f} [{expected_min:.1f}%, {expected_max:.1f}%]     {actual_change:<12.2f} {'✓' if is_valid else '✗':<6}")
            
            if total_samples > 10:
                print("...")
            
            violation_rate = (constraint_violations / total_samples) * 100
            print(f"\nSamples violating 1.5x-2.3x magnitude constraint: {constraint_violations}/{total_samples} ({violation_rate:.1f}%)")
        
        print(f"\n📊 Target Statistics:")
        print(f"  Mean percentage change: {y_all.mean():.2f}%")
        print(f"  Std percentage change: {y_all.std():.2f}%")
        print(f"  Min percentage change: {y_all.min():.2f}%")
        print(f"  Max percentage change: {y_all.max():.2f}%")
        
        # Check if target values are mostly positive (as expected)
        positive_count = (y_all > 0).sum()
        positive_rate = (positive_count / len(y_all)) * 100
        print(f"  Positive changes: {positive_count}/{len(y_all)} ({positive_rate:.1f}%)")
        
        if positive_rate < 80:
            print(f"  ⚠️  Warning: Only {positive_rate:.1f}% of targets are positive. Expected mostly positive rises!")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state)
        print(f"\n📊 Dataset Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples : {len(X_test)}")
        print(f"  Total features: {len(ordered_features)}")
        
        # Lightweight model configurations with regularization
        models_to_try = {
            'regularized_rf': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('rf', RandomForestRegressor(random_state=random_state, n_jobs=-1))
                ]),
                'params': {
                    'rf__n_estimators': [30,50,70],  # Reduced options
                    'rf__max_depth': [6, 8,12],  # Just 2 options
                    'rf__min_samples_split': [2,5, 10],  # Just 2 options
                    'rf__min_samples_leaf': [2, 4],  # Just 2 options
                    'rf__max_features': [0.7, 0.9, None]  # Just 2 options
                }
            },
            'lasso_reg': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('lasso', Lasso(random_state=random_state, max_iter=1000))
                ]),
                'params': {
                    'lasso__alpha': [0.1, 0.5, 1.0 ,3.0]  # Just 4 regularization strengths
                }
            }
        }
        
        best_model = None
        best_score = np.inf
        best_model_name = None
        
        for model_name, config in models_to_try.items():
            print(f"\n🔍 Testing {model_name}...")
            
            kfold = KFold(n_splits=3, shuffle=True, random_state=random_state)  # Reduced to 3 folds
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=kfold, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred_test = grid_search.predict(X_test)
            y_pred_train = grid_search.predict(X_train)
            
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            overfitting_ratio = test_rmse / train_rmse
            
            print(f"  {model_name}:")
            print(f"    Train RMSE: {train_rmse:.2f}%")
            print(f"    Test RMSE:  {test_rmse:.2f}%")
            print(f"    Overfitting ratio: {overfitting_ratio:.3f}")
            
            if test_rmse < best_score:
                best_score = test_rmse
                best_model = grid_search.best_estimator_
                best_model_name = model_name
        
        print(f"\n✅ Best model: {best_model_name}")
        
        # Final evaluation
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        overfitting_ratio = test_rmse / train_rmse
        
        print("\n📈 Final Model Performance:")
        print(f"  Training: RMSE = {train_rmse:.2f}%, MAE = {train_mae:.2f}%")
        print(f"  Testing:  RMSE = {test_rmse:.2f}%, MAE = {test_mae:.2f}%")
        print(f"  Overfitting ratio: {overfitting_ratio:.3f} {'✅ Good' if overfitting_ratio < 1.1 else '⚠️ Some overfitting' if overfitting_ratio < 1.2 else '❌ Significant overfitting'}")
        print(f"  🎯 Note: Runtime predictions constrained to POSITIVE rises (1.5x-2.3x initial drop magnitude)")
        
        # Feature importance (only for Random Forest)
        if hasattr(best_model.named_steps.get('rf', None), 'feature_importances_'):
            feature_importance = best_model.named_steps['rf'].feature_importances_
            
            feat_imp_df = pd.DataFrame({
                'Feature': ordered_features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            print("\n🔑 Feature Importances (Top 10):")
            print(feat_imp_df.head(10))
            
            # Check if critical features are highly ranked
            for feature in available_critical:
                if feature in feat_imp_df['Feature'].values:
                    rank = feat_imp_df[feat_imp_df['Feature'] == feature].index[0] + 1
                    importance = feat_imp_df[feat_imp_df['Feature'] == feature]['Importance'].values[0]
                    print(f"🔥 {feature}: Rank #{rank}, Importance = {importance:.4f}")
            
        else:
            # Fallback for Lasso (no feature importances)
            feat_imp_df = pd.DataFrame({
                'Feature': ordered_features,
                'Importance': np.ones(len(ordered_features)) / len(ordered_features)
            })
            print("\n🔍 Lasso selected - feature importances not available")
        
        return {
            'model': best_model,
            'selected_features': ordered_features,
            'feature_importance': feat_imp_df,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'model_type': best_model_name,
                'overfitting_ratio': overfitting_ratio
            }
        }
class ABAAController(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            print("ABAAController initializing...")
            
            # Initialize class variables
            self.measurement_thread = None
            self.baking_thread = None
            self.training_thread = None
            self.current_wafer_dir = None
            self.current_wafer_name = None
            self.current_model_path = None
            self.last_resistance = None
            self.last_prediction = None
            self.resistance_text_item = None
            self.prediction_text_item = None
            self.error_bar_item = None
            
            # Add timestamp storage
            self.baking_end_timestamp = None
            self.cooled_start_timestamp = None
            self.final_baking_resistance = None
            
            # Initialize the user and wafer directories
            self.current_user_dir = None
            
            # Set up the main window
            self.setWindowTitle("ABAA Smart Controller")
            self.setGeometry(100, 100, 1200, 800)
            
            # Ensure window is resizable (default behavior)
            self.setWindowFlags(Qt.Window)
            
            # Set application-wide font sizes
            self.setStyleSheet(f"""
                QWidget {{ font-size: {BASE_FONT_SIZE}px; }}
                QLabel {{ font-size: {LABEL_FONT_SIZE}px; }}
                QPushButton {{ font-size: {BUTTON_FONT_SIZE}px; padding: 10px; }}
                QGroupBox {{ font-size: {BASE_FONT_SIZE}px; font-weight: bold; }}
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ font-size: {BASE_FONT_SIZE}px; padding: 8px; }}
            """)
            # Enhance the stylesheet with more professional styling
            self.setStyleSheet(self.styleSheet() + """
                QMainWindow {
                    background-color: #f5f5f5;
                }
                
                QGroupBox {
                    border: 2px solid #d1d1d1;
                    border-radius: 8px;
                    margin-top: 1.5ex;
                    background-color: white;
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    background-color: white;
                }
                
                QPushButton {
                    background-color: #0275d8;
                    color: white;
                    border-radius: 5px;
                }
                
                QPushButton:hover {
                    background-color: #0069d9;
                }
                
                QPushButton:disabled {
                    background-color: #6c757d;
                }
                
                QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    background-color: white;
                }
                
                QHeaderView::section {
                    background-color: #0275d8;
                    color: white;
                    padding: 5px;
                    border: 1px solid #0275d8;
                }
                
                QTableWidget {
                    alternate-background-color: #f9f9f9;
                    selection-background-color: #007bff;
                }
                
                QTabWidget::pane {
                    border: 1px solid #ddd;
                    background: white;
                }
                
                QTabBar::tab {
                    background: #f8f9fa;
                    border: 1px solid #ddd;
                    padding: 6px 12px;
                    margin-right: 2px;
                }
                
                QTabBar::tab:selected {
                    background: white;
                    border-bottom-color: white;
                }
                
                QFrame {
                    border-radius: 3px;
                }
            """)
            # Create the central widget and layout
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            self.main_layout = QVBoxLayout(self.central_widget)
            
            # Create the main screen with all functionality
            try:
                self.create_main_screen()
                print("Main screen created")
            except Exception as e:
                print(f"Error creating main screen: {e}")
                import traceback
                traceback.print_exc()
            
            # Initialize data directory
            try:
                self.init_data_directory()
                print("Data directory initialized")
            except Exception as e:
                print(f"Error initializing data directory: {e}")
            
            # Check hardware connection
            try:
                self.check_hardware_connection()
                print("Hardware connection checked")
            except Exception as e:
                print(f"Error checking hardware connection: {e}")
                QMessageBox.critical(self, "Hardware Error", 
                                     "Could not connect to hardware. Please check connections and restart the application.")
                sys.exit(1)
            
            print("ABAAController initialization complete")
                
        except Exception as e:
            print(f"Error in ABAAController initialization: {e}")
            import traceback
            traceback.print_exc()
            
    def check_hardware_connection(self):
        """Check if the hardware is connected and enter demo mode on failure"""
        global HARDWARE_CONNECTED
        try:
            rm = pyvisa.ResourceManager()
            keithley = rm.open_resource(KEITHLEY_RESOURCE_NAME)
            keithley.write("*IDN?")
            idn = keithley.read()
            keithley.close()

            # Hardware detected
            self.statusBar().showMessage(f"Connected to: {idn.strip()}")
            HARDWARE_CONNECTED = True
            return True

        except Exception as e:
            # Enter demo mode
            HARDWARE_CONNECTED = False
            self.statusBar().showMessage("‼️ Running in demo mode: measurements are simulated.")
            QMessageBox.information(
                self,
                "Demo Mode",
                "Hardware not found. Running in demo mode; measurements will be simulated."
            )
            return False
    
            
    def shutdown_hardware(self):
        """Safely shut down hardware connections"""
        try:
            rm = pyvisa.ResourceManager()
            keithley = rm.open_resource(KEITHLEY_RESOURCE_NAME)
            
            # Turn off output
            keithley.write(":OUTP OFF")
            keithley.close()
            print("Hardware connection closed safely")
        except Exception as e:
            print(f"Error during hardware shutdown: {str(e)}")
                
    def closeEvent(self, event):
        """Handle window close event to safely shut down hardware"""
        # Stop any running operations
        if self.measurement_thread and self.measurement_thread.running:
            self.toggle_measurement()
        if self.baking_thread and self.baking_thread.running:
            self.toggle_baking()
        
        # Shutdown hardware
        self.shutdown_hardware()
        
        # Accept the close event
        event.accept()
    
    def init_data_directory(self):
        """Initialize the data directory structure if it doesn't exist"""
        # Create base directory
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        
        # Create users directory
        if not os.path.exists(USERS_DIR):
            os.makedirs(USERS_DIR)
    
    def create_main_screen(self):
        """Create a unified main screen with the baking process interface"""
        self.main_widget = QWidget()
        main_layout = QVBoxLayout(self.main_widget)
        
        # Create header with emoji
        header_layout = QHBoxLayout()
        
        # Title with emojis
        title_label = QLabel("⚡ ABAA Smart Controller ⚡")
        title_label.setFont(QFont("Arial", int(100 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label, 1)
        
        # Hardware status with attribution underneath
        right_section = QWidget()
        right_layout = QVBoxLayout(right_section)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)  # Minimal spacing
        
        hardware_box = QGroupBox("Hardware Status")
        hardware_box.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        hardware_layout = QVBoxLayout()
        
        self.hw_status_label = QLabel("Hardware Connected")
        self.hw_status_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        self.hw_status_label.setAlignment(Qt.AlignCenter)
        self.hw_status_label.setStyleSheet("color: green; font-weight: bold;")
        
        hardware_layout.addWidget(self.hw_status_label)
        hardware_box.setLayout(hardware_layout)
        right_layout.addWidget(hardware_box)
        
        # Add attribution text under hardware status
        attribution_label = QLabel("Created by Dany and Noam")
        attribution_label.setFont(QFont("Arial", int(6 * FONT_SIZE_MULTIPLIER)))  # Smaller font
        attribution_label.setAlignment(Qt.AlignRight)
        attribution_label.setStyleSheet("color: #AAAAAA;")  # Lighter gray color
        right_layout.addWidget(attribution_label)
        
        header_layout.addWidget(right_section)
        
        main_layout.addLayout(header_layout)
        
        # Main content splitter
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel - ALL CONTROLS
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        self.create_left_panel_controls(left_layout)
        
        # Right panel - ONLY DISPLAYS AND PLOTS  
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        self.create_right_panel_displays(right_layout)
        
        content_splitter.addWidget(self.left_panel)
        content_splitter.addWidget(self.right_panel)
        
        # Set sizes: Left 30%, Right 70%
        content_splitter.setSizes([360, 840])
        content_splitter.setStretchFactor(0, 0)  # Left panel fixed
        content_splitter.setStretchFactor(1, 1)  # Right panel stretches
        
        main_layout.addWidget(content_splitter, 1)
        
        # Status bar at the bottom
        self.status_label = QLabel("Ready - Select a user and wafer to begin")
        self.status_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)
        
        # Add the main widget to the main layout
        self.main_layout.addWidget(self.main_widget)

    def create_left_panel_controls(self, parent_layout):
        """Create all controls in the left panel with collapsible sections"""
        
        # Data Selection (collapsible)
        self.data_selection_group = QGroupBox("⚙️ Data Selection")
        self.data_selection_group.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.data_selection_group.setCheckable(True)
        self.data_selection_group.setChecked(True)
        
        data_layout = QVBoxLayout()
        
        # User selection
        user_layout = QHBoxLayout()
        self.user_label = QLabel("User: Not Selected")
        self.user_label.setFont(QFont("Arial", BASE_FONT_SIZE, QFont.Bold))
        user_layout.addWidget(self.user_label, 1)
        
        select_user_btn = QPushButton("Select User...")
        select_user_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        select_user_btn.clicked.connect(self.select_user)
        user_layout.addWidget(select_user_btn)
        data_layout.addLayout(user_layout)
        
        # Wafer selection
        wafer_layout = QHBoxLayout()
        self.wafer_label = QLabel("Wafer: Not Selected")
        self.wafer_label.setFont(QFont("Arial", BASE_FONT_SIZE, QFont.Bold))
        wafer_layout.addWidget(self.wafer_label, 1)
        
        select_wafer_btn = QPushButton("Select Wafer...")
        select_wafer_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        select_wafer_btn.clicked.connect(self.on_select_wafer_clicked)
        wafer_layout.addWidget(select_wafer_btn)
        data_layout.addLayout(wafer_layout)
        
        self.data_selection_group.setLayout(data_layout)
        parent_layout.addWidget(self.data_selection_group)
        
        # Model Selection (collapsible)
        self.model_group = QGroupBox("⚙️ Model Selection")
        self.model_group.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.model_group.setCheckable(True)
        self.model_group.setChecked(True)
        
        model_layout = QVBoxLayout()
        
        # Current model display
        model_info_layout = QHBoxLayout()
        model_info_layout.addWidget(QLabel("Current Model:"))
        self.current_model_label = QLabel("None Selected")
        self.current_model_label.setFont(QFont("Arial", BASE_FONT_SIZE))
        model_info_layout.addWidget(self.current_model_label, 1)
        model_layout.addLayout(model_info_layout)
        
        # Model buttons
        model_btn_layout = QHBoxLayout()
        self.select_model_btn = QPushButton("Select Model...")
        self.select_model_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.select_model_btn.clicked.connect(self.show_model_selection)
        model_btn_layout.addWidget(self.select_model_btn)
        
        model_info_btn = QPushButton("Model Info")
        model_info_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        model_info_btn.clicked.connect(self.show_model_info)
        model_btn_layout.addWidget(model_info_btn)
        model_layout.addLayout(model_btn_layout)
        
        self.model_group.setLayout(model_layout)
        parent_layout.addWidget(self.model_group)
        
        # Wafer Status (compact, always visible)
        status_section = QGroupBox("Wafer Status")
        status_section.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("Untrained:"), 0, 0)
        self.untrained_count_label = QLabel("0")
        self.untrained_count_label.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.untrained_count_label.setStyleSheet("color: blue;")
        status_layout.addWidget(self.untrained_count_label, 0, 1)
        
        status_layout.addWidget(QLabel("Trained:"), 0, 2)
        self.trained_count_label = QLabel("0")
        self.trained_count_label.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.trained_count_label.setStyleSheet("color: green;")
        status_layout.addWidget(self.trained_count_label, 0, 3)
        
        self.retrain_model_btn = QPushButton("Retrain Model")
        self.retrain_model_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.retrain_model_btn.clicked.connect(self.retrain_model)
        self.retrain_model_btn.setEnabled(False)
        status_layout.addWidget(self.retrain_model_btn, 1, 0, 1, 4)
        
        status_section.setLayout(status_layout)
        parent_layout.addWidget(status_section)
        
        # Baking Settings (collapsible, collapsed by default)
        self.baking_settings_group = QGroupBox("⚙️ Baking Settings")
        self.baking_settings_group.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.baking_settings_group.setCheckable(True)
        self.baking_settings_group.setChecked(False)  # Collapsed by default
        
        settings_layout = QGridLayout()
        
        # Chip identification
        settings_layout.addWidget(QLabel("Wafer:"), 0, 0)
        self.baking_wafer_label = QLabel("Not Selected")
        self.baking_wafer_label.setFont(QFont("Arial", BASE_FONT_SIZE, QFont.Bold))
        settings_layout.addWidget(self.baking_wafer_label, 0, 1)
        
        settings_layout.addWidget(QLabel("Row:"), 1, 0)
        self.baking_row_input = QSpinBox()
        self.baking_row_input.setMinimum(1)
        self.baking_row_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.baking_row_input, 1, 1)
        
        settings_layout.addWidget(QLabel("Chip:"), 2, 0)
        self.baking_chip_input = QLineEdit("1")
        self.baking_chip_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.baking_chip_input, 2, 1)
        
        settings_layout.addWidget(QLabel("Temperature (°C):"), 3, 0)
        self.baking_temp_input = QDoubleSpinBox()
        self.baking_temp_input.setMinimum(0)
        self.baking_temp_input.setMaximum(500)
        self.baking_temp_input.setValue(100)
        self.baking_temp_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.baking_temp_input, 3, 1)
        
        # Baking parameters
        settings_layout.addWidget(QLabel("Voltage (V):"), 4, 0)
        self.voltage_input = QDoubleSpinBox()
        self.voltage_input.setMinimum(0.1)
        self.voltage_input.setMaximum(2.0)
        self.voltage_input.setValue(0.8)
        self.voltage_input.setSingleStep(0.1)
        self.voltage_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.voltage_input, 4, 1)
        
        settings_layout.addWidget(QLabel("Frequency (Hz):"), 5, 0)
        self.frequency_input = QDoubleSpinBox()
        self.frequency_input.setMinimum(0.1)
        self.frequency_input.setMaximum(5.0)
        self.frequency_input.setValue(1.0)
        self.frequency_input.setSingleStep(0.1)
        self.frequency_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.frequency_input, 5, 1)
        
        settings_layout.addWidget(QLabel("Current Limit:"), 6, 0)
        self.current_limit_dropdown = QComboBox()
        self.current_limit_dropdown.addItems(["1 µA", "10 µA", "100 µA", "200 µA", "300 µA", "400 µA", "600 µA", "800 µA"])
        self.current_limit_dropdown.setCurrentIndex(7)
        self.current_limit_dropdown.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.current_limit_dropdown, 6, 1)
        
        settings_layout.addWidget(QLabel("Target Resistance (Ω):"), 7, 0)
        self.target_resistance_input = QDoubleSpinBox()
        self.target_resistance_input.setMinimum(0)
        self.target_resistance_input.setMaximum(100000)
        self.target_resistance_input.setValue(6000)
        self.target_resistance_input.valueChanged.connect(self.update_target_display)
        self.target_resistance_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.target_resistance_input, 7, 1)
        
        self.auto_stop_checkbox = QCheckBox("Auto-stop at target")
        self.auto_stop_checkbox.setFont(QFont("Arial", BASE_FONT_SIZE))
        self.auto_stop_checkbox.setChecked(True) 
        settings_layout.addWidget(self.auto_stop_checkbox, 8, 0, 1, 2)
        
        settings_layout.addWidget(QLabel("Prediction Offset (Ω):"), 9, 0)
        self.offset_input = QDoubleSpinBox()
        self.offset_input.setMinimum(-5000)
        self.offset_input.setMaximum(5000)
        self.offset_input.setValue(0)
        self.offset_input.setSingleStep(10)
        self.offset_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        settings_layout.addWidget(self.offset_input, 9, 1)
        
        self.baking_settings_group.setLayout(settings_layout)
        parent_layout.addWidget(self.baking_settings_group)
        
        # Action Buttons with emojis
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.quick_measure_btn = QPushButton("🔬 Quick Measurement")
        self.quick_measure_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.quick_measure_btn.clicked.connect(self.toggle_measurement)
        buttons_layout.addWidget(self.quick_measure_btn)
        
        self.baking_start_btn = QPushButton("🔥 Start Baking")
        self.baking_start_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.baking_start_btn.clicked.connect(self.toggle_baking)
        buttons_layout.addWidget(self.baking_start_btn)
        
        button_row = QHBoxLayout()
        self.baking_clear_btn = QPushButton("Clear")
        self.baking_clear_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.baking_clear_btn.clicked.connect(self.clear_baking_data)
        button_row.addWidget(self.baking_clear_btn)
        
        self.baking_save_btn = QPushButton("💾 Save Data")
        self.baking_save_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.baking_save_btn.clicked.connect(self.save_baking_data)
        self.baking_save_btn.setEnabled(False)
        button_row.addWidget(self.baking_save_btn)
        
        buttons_layout.addLayout(button_row)
        parent_layout.addLayout(buttons_layout)
        
        # Removed the status text from bottom left completely
        
        # Add stretch at the bottom
        parent_layout.addStretch()
    def create_right_panel_displays(self, parent_layout):
        """Create the right panel with only measurement displays and graphs"""
        
        # Measurement Values Display
        self.measurement_display = self.create_measurement_display()
        parent_layout.addWidget(self.measurement_display)
        
        # Create the graphs
        self.create_graphs_panel(parent_layout)

    def create_measurement_display(self):
        """Create a widget to display current measurement and prediction values"""
        display_widget = QGroupBox("Measurement Values")
        display_widget.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))

        layout = QHBoxLayout()

        # Current measurement display
        current_frame = QFrame()
        current_frame.setFrameShape(QFrame.Box)
        current_frame.setStyleSheet(
            "background-color: #f8f9fa; border: 2px solid #dee2e6; \
             border-radius: 10px; padding: 15px;"
        )
        current_layout = QVBoxLayout(current_frame)

        current_title = QLabel("Current Resistance")
        current_title.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        current_title.setAlignment(Qt.AlignCenter)
        current_title.setStyleSheet("color: #495057;")
        current_layout.addWidget(current_title)

        self.current_resistance_label = QLabel("--- Ω")
        self.current_resistance_label.setFont(QFont("Arial", int(35 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.current_resistance_label.setAlignment(Qt.AlignCenter)
        self.current_resistance_label.setStyleSheet("color: #d9534f; margin: 5px 0;")
        current_layout.addWidget(self.current_resistance_label)

        layout.addWidget(current_frame)

        # Predicted measurement display (error label removed)
        predicted_frame = QFrame()
        predicted_frame.setFrameShape(QFrame.Box)
        predicted_frame.setStyleSheet(
            "background-color: #f8f9fa; border: 2px solid #dee2e6; \
             border-radius: 10px; padding: 15px;"
        )
        predicted_layout = QVBoxLayout(predicted_frame)

        predicted_title = QLabel("Predicted Resistance")
        predicted_title.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        predicted_title.setAlignment(Qt.AlignCenter)
        predicted_title.setStyleSheet("color: #495057;")
        predicted_layout.addWidget(predicted_title)

        self.predicted_resistance_label = QLabel("--- Ω")
        self.predicted_resistance_label.setFont(QFont("Arial", int(26 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.predicted_resistance_label.setAlignment(Qt.AlignCenter)
        self.predicted_resistance_label.setStyleSheet("color: #5cb85c; margin: 5px 0;")
        predicted_layout.addWidget(self.predicted_resistance_label)

        layout.addWidget(predicted_frame)

        # Change display (unchanged)
        change_frame = QFrame()
        change_frame.setFrameShape(QFrame.Box)
        change_frame.setStyleSheet(
            "background-color: #f8f9fa; border: 2px solid #dee2e6; \
             border-radius: 10px; padding: 15px;"
        )
        change_layout = QVBoxLayout(change_frame)

        change_title = QLabel("Change from Initial")
        change_title.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        change_title.setAlignment(Qt.AlignCenter)
        change_title.setStyleSheet("color: #495057;")
        change_layout.addWidget(change_title)

        self.change_display_label = QLabel("--- Ω (---%)")
        self.change_display_label.setFont(QFont("Arial", int(26 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.change_display_label.setAlignment(Qt.AlignCenter)
        self.change_display_label.setStyleSheet("color: #0275d8; margin: 5px 0;")
        change_layout.addWidget(self.change_display_label)

        layout.addWidget(change_frame)

        display_widget.setLayout(layout)
        return display_widget


    def create_graphs_panel(self, parent_layout):
        """Create the graphs panel (unchanged)"""
        # Define plot style variables
        label_style = {'font-size': f'{int(9 * FONT_SIZE_MULTIPLIER)}pt'}
        axis_style = {'tickFont': QFont('Arial', int(7 * FONT_SIZE_MULTIPLIER))}
        
        # Create resistance plot (takes 2/3 of vertical space)
        self.baking_plot = pg.PlotWidget()
        self.baking_plot.setBackground('w')
        
        self.baking_plot.setLabel('left', 'Resistance', units='Ω')
        self.baking_plot.getAxis('left').setScale(1.0)
        self.baking_plot.getAxis('left').enableAutoSIPrefix(False)
        
        def resistance_tick_formatter(values):
            return [f"{val:.2f}" for val in values]

        self.baking_plot.getAxis('left').setTickStrings = resistance_tick_formatter
        self.baking_plot.setLabel('bottom', 'Time', units='s')
        
        self.baking_plot.getAxis('left').setLabel('Resistance', units='Ω', **label_style)
        self.baking_plot.getAxis('bottom').setLabel('Time', units='s', **label_style)
        self.baking_plot.getAxis('left').setStyle(**axis_style)
        self.baking_plot.getAxis('bottom').setStyle(**axis_style)
        
        self.baking_plot.showGrid(x=True, y=True)
        
        # Create scatter plot for resistance data 
        self.baking_resistance_curve = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 0, 0), pen=None, name='Resistance'
        )
        self.baking_plot.addItem(self.baking_resistance_curve)
        
        # Create prediction point
        self.predicted_resistance_point = pg.ScatterPlotItem(
            size=15, brush=pg.mkBrush(0, 0, 255), pen=pg.mkPen(color='b', width=2), name='Predicted'
        )
        self.baking_plot.addItem(self.predicted_resistance_point)
        
        # Add error bar item (manual drawing)
        self.error_bar_item = None
        
        # Add target resistance line
        self.target_line = pg.InfiniteLine(
            angle=0, movable=False, 
            pen=pg.mkPen(color='g', width=3, style=Qt.DashLine),
            label='Target: {value:.2f} Ω', labelOpts={'position': 0.9}
        )
        self.baking_plot.addItem(self.target_line)
        
        # Add legend
        self.baking_plot.addLegend()
        
        # Create voltage plot
        self.voltage_plot = pg.PlotWidget()
        self.voltage_plot.setBackground('w')
        self.voltage_plot.setLabel('left', 'Voltage', units='V')
        self.voltage_plot.setLabel('bottom', 'Time', units='s')
        
        self.voltage_plot.getAxis('left').setLabel('Voltage', units='V', **label_style)
        self.voltage_plot.getAxis('bottom').setLabel('Time', units='s', **label_style)
        self.voltage_plot.getAxis('left').setStyle(**axis_style)
        self.voltage_plot.getAxis('bottom').setStyle(**axis_style)
        
        self.voltage_plot.showGrid(x=True, y=True)
        
        # Create scatter plot for voltage data
        self.voltage_curve = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush(0, 0, 255), pen=None, name='Voltage'
        )
        self.voltage_plot.addItem(self.voltage_curve)
        
        # Add plots to layout
        parent_layout.addWidget(self.baking_plot, 2)  # Resistance gets 2/3 of space
        parent_layout.addWidget(self.voltage_plot, 1)  # Voltage gets 1/3 of space
        
        # Set initial target line
        self.target_line.setValue(6000)  # Default target of 6000 ohms

    # NOTE: The old create_controls_panel() function should be completely REMOVED 
    # from your code as all its functionality has been moved to create_left_panel_controls()
    def update_target_display(self, value):
        """Update the target resistance display"""
        # Update the target line on the plot
        self.target_line.setValue(value)
        self.target_line.label.setFormat(f"Target: {value:.2f} Ω")
        
        # Note: target_resistance_display widget doesn't exist in the current implementation,
        # so we should not reference it here.
        # self.target_resistance_display.setText(f"{value:.2f} Ω")  # This line causes the crash
    def retrain_model(self):
        """Start the model training process with preprocessing"""
        if not self.current_wafer_dir:
            QMessageBox.warning(self, "No Wafer Selected", "Please select a wafer first.")
            return
            
        # Check if there are untrained files
        untrained_dir = os.path.join(self.current_wafer_dir, UNTRAINED_DIR)
        # if not os.path.exists(untrained_dir):
        #     QMessageBox.warning(self, "No Untrained Data", "There are no untrained files to train on.")
        #     return
            
        untrained_files = [f for f in os.listdir(untrained_dir) if f.endswith('.csv')]
        # if not untrained_files:
        #     QMessageBox.warning(self, "No Untrained Files", "There are no untrained files to train on.")
        #     return
        
        # Create progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Training Model")
        progress_dialog.setFixedSize(500, 200)
        # Remove question mark button and add resizing
        progress_dialog.setWindowFlags(progress_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        dialog_layout = QVBoxLayout(progress_dialog)
        
        # Add title
        title = QLabel("Training New Model")
        title.setFont(QFont("Arial", int(16 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(title)
        
        # Add progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(True)
        progress_bar.setFont(QFont("Arial", BASE_FONT_SIZE))
        dialog_layout.addWidget(progress_bar)
        
        # Status message
        status_label = QLabel("Starting preprocessing...")
        status_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        status_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        button_layout.addWidget(cancel_btn)
        dialog_layout.addLayout(button_layout)
        
        # Show the dialog but don't block
        progress_dialog.setModal(False)
        progress_dialog.show()
        
        # First, import the PreprocessingThread from the preprocess_single_csv module
        try:
            # Make sure the file is properly imported
            import sys
            
            # Add the current directory to the path if needed
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
                
            # Now import the PreprocessingThread class
            from preprocess_single_csv import PreprocessingThread
            
            # Create a customized preprocessing thread that forwards progress to our UI
            class UIPreprocessingThread(PreprocessingThread, QThread):
                progress_update_signal = pyqtSignal(int, str)
                preprocessing_complete_signal = pyqtSignal()
                error_occurred_signal = pyqtSignal(str)
                
                def __init__(self, wafer_dir):
                    PreprocessingThread.__init__(self, wafer_dir)
                    QThread.__init__(self)
                    
                def progress_update(self, percentage, message):
                    self.progress_update_signal.emit(percentage, message)
                    print(f"Preprocessing progress: {percentage}% - {message}")
                    
                def preprocessing_complete(self):
                    self.preprocessing_complete_signal.emit()
                    print("Preprocessing complete")
                    
                def error_occurred(self, message):
                    self.error_occurred_signal.emit(message)
                    print(f"Preprocessing error: {message}")
                    
                def run(self):
                    PreprocessingThread.run(self)
            
            # Create the preprocessing thread
            preprocess_thread = UIPreprocessingThread(self.current_wafer_dir)
            
            # Connect signals
            preprocess_thread.progress_update_signal.connect(
                lambda p, m: self.update_training_progress(progress_bar, status_label, p, m)
            )
            preprocess_thread.preprocessing_complete_signal.connect(
                lambda: self.continue_with_model_training(progress_dialog, progress_bar, status_label)
            )
            preprocess_thread.error_occurred_signal.connect(
                lambda msg: self.on_training_error(progress_dialog, msg)
            )
            
            # Connect cancel button
            cancel_btn.clicked.connect(progress_dialog.reject)
            
            # Start preprocessing
            preprocess_thread.start()
            
            # Store thread reference to prevent garbage collection
            self.preprocess_thread = preprocess_thread
            
        except ImportError as e:
            error_msg = f"Could not import preprocessing module: {str(e)}"
            QMessageBox.critical(self, "Import Error", error_msg)
            self.on_training_error(progress_dialog, error_msg)
        except Exception as e:
            error_msg = f"Error initializing preprocessing: {str(e)}"
            QMessageBox.critical(self, "Preprocessing Error", error_msg)
            self.on_training_error(progress_dialog, error_msg)

    def continue_with_model_training(self, progress_dialog, progress_bar, status_label):
        """Continue with model training after preprocessing is complete"""
        try:
            # Update progress
            self.update_training_progress(progress_bar, status_label, 50, "Preprocessing complete. Starting model training...")
            
            # Create and start the training thread
            self.training_thread = ModelTrainingThread(
                self.current_wafer_dir, 
                self.current_wafer_name
            )
            
            # Connect signals
            self.training_thread.progress_update.connect(
                lambda p, m: self.update_training_progress(progress_bar, status_label, 50 + p//2, m)
            )
            self.training_thread.training_complete.connect(
                lambda path: self.on_training_complete(progress_dialog, path)
            )
            self.training_thread.error_occurred.connect(
                lambda msg: self.on_training_error(progress_dialog, msg)
            )
            
            # Start training
            self.training_thread.start()
        except Exception as e:
            self.on_training_error(progress_dialog, f"Error starting model training: {str(e)}")

    def update_training_progress(self, progress_bar, status_label, percentage, message):
        """Update the training progress dialog"""
        progress_bar.setValue(percentage)
        status_label.setText(message)

    def on_training_complete(self, dialog, model_path):
        """Handle completion of model training"""
        # Update dialog
        dialog.setWindowTitle("Training Complete")
        
        # Get the model filename
        model_filename = os.path.basename(model_path)
        
        # Load the new model
        self.current_model_path = model_path
        self.current_model_label.setText(model_filename)
        
        # Update UI
        self.update_untrained_count()  # This will show 0 untrained files now
        self.update_file_counts()  # Update both trained and untrained counts
        
        # Show success message
        QMessageBox.information(
            self, 
            "Training Complete", 
            f"New model '{model_filename}' has been trained successfully.\n\n"
            f"This model is now selected for prediction."
        )
        
        # Close the progress dialog
        dialog.accept()

    def on_training_error(self, dialog, error_message):
        """Handle errors during model training"""
        # Update dialog
        dialog.setWindowTitle("Training Error")
        
        # Show error message
        QMessageBox.critical(
            self, 
            "Training Error", 
            f"An error occurred during model training:\n\n{error_message}"
        )
        
        # Close the progress dialog
        dialog.reject()

    def refresh_wafer_list(self):
        """Refresh the dropdown list of available wafers"""
        print("Refreshing wafer list...")
        # If we have a wafer selection dropdown, refresh it
        if hasattr(self, 'wafer_selection_dropdown'):
            try:
                self.wafer_selection_dropdown.clear()
                
                # Check if DATA_DIR exists
                if os.path.exists(DATA_DIR):
                    # Get all directories (potential wafers)
                    wafers = [d for d in os.listdir(DATA_DIR) 
                            if os.path.isdir(os.path.join(DATA_DIR, d))]
                    
                    if wafers:
                        self.wafer_selection_dropdown.addItems(wafers)
                        # Select current wafer if it exists
                        if self.current_wafer_name:
                            index = self.wafer_selection_dropdown.findText(self.current_wafer_name)
                            if index >= 0:
                                self.wafer_selection_dropdown.setCurrentIndex(index)
                    else:
                        self.wafer_selection_dropdown.addItem("No wafers available")
                else:
                    self.wafer_selection_dropdown.addItem("No wafers available")
            except Exception as e:
                print(f"Error in refresh_wafer_list: {e}")
        
        # Update counts if necessary
        if hasattr(self, 'update_file_counts'):
            try:
                self.update_file_counts()
            except Exception as e:
                print(f"Error updating file counts: {e}")

    def update_file_counts(self):
        """Update both trained and untrained file counts"""
        self.update_untrained_count()
        
        # Update trained count if we have a current wafer
        if self.current_wafer_dir:
            trained_dir = os.path.join(self.current_wafer_dir, TRAINED_DIR)
            if os.path.exists(trained_dir):
                trained_files = [f for f in os.listdir(trained_dir) if f.endswith('.csv')]
                self.trained_count_label.setText(str(len(trained_files)))
            else:
                self.trained_count_label.setText("0")
        else:
            self.trained_count_label.setText("0")
            
    def select_user(self):
        """Show file browser to select an existing user or create a new one"""
        # Open file browser starting at USERS_DIR
        selected_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select User Folder (or create new folder for new user)", 
            USERS_DIR,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not selected_dir:
            return  # User cancelled
        
        # Normalize paths to handle different separators and cases
        normalized_selected = os.path.normpath(selected_dir).lower()
        normalized_users_dir = os.path.normpath(USERS_DIR).lower()
        
        # Check if the selected directory is within USERS_DIR
        if not normalized_selected.startswith(normalized_users_dir):
            QMessageBox.warning(
                self, 
                "Invalid Location", 
                f"Please select a folder within the users directory:\n{USERS_DIR}\n\n"
                f"Selected: {selected_dir}"
            )
            return
        
        # Get the user name from the directory path
        user_name = os.path.basename(selected_dir)
        
        if not user_name:
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid user folder.")
            return
        
        # Store the user directory (use original path, not normalized)
        self.current_user_dir = selected_dir
        
        # Updated required folders list
        required_folders = ['models', 'trained', 'untrained', 'dead', 'quick_measure']
        existing_folders = []
        missing_folders = []
        
        # Check what already exists
        if os.path.exists(selected_dir):
            existing_items = os.listdir(selected_dir)
            for folder in required_folders:
                folder_path = os.path.join(selected_dir, folder)
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    existing_folders.append(folder)
                else:
                    missing_folders.append(folder)
        else:
            # Directory doesn't exist, create it
            os.makedirs(selected_dir)
            missing_folders = required_folders
        
        # If missing folders, create them
        if missing_folders:
            for folder in missing_folders:
                folder_path = os.path.join(selected_dir, folder)
                os.makedirs(folder_path)
            
            # Show confirmation message
            if len(missing_folders) == len(required_folders):
                # Completely new user
                QMessageBox.information(
                    self, 
                    "New User Created", 
                    f"New user '{user_name}' has been created with the following structure:\n\n"
                    f"• models (for .pkl model files)\n"
                    f"• trained (for processed data files)\n"
                    f"• untrained (for new data files)\n"
                    f"• dead (for archived files)\n\n"
                    f"Location: {selected_dir}"
                )
            else:
                # Existing user with missing folders
                missing_list = '\n• '.join(missing_folders)
                QMessageBox.information(
                    self, 
                    "User Structure Updated", 
                    f"User '{user_name}' structure has been updated.\n\n"
                    f"Created missing folders:\n• {missing_list}\n\n"
                    f"Location: {selected_dir}"
                )
        
        # Update UI
        self.user_label.setText(f"User: {user_name}")
        
        # Reset wafer selection since we changed users
        self.current_wafer_dir = None
        self.current_wafer_name = None
        self.wafer_label.setText("Wafer: Not Selected")
        if hasattr(self, 'baking_wafer_label'):
            self.baking_wafer_label.setText("Not Selected")
        
        # Reset model selection
        self.current_model_path = None
        self.current_model_label.setText("None Selected")
        
        # Update file counts (will be 0 until wafer is selected)
        self.update_file_counts()
        
        # Update status
        self.status_label.setText(f"User '{user_name}' selected. Please select a wafer to continue.")
        
        print(f"User '{user_name}' selected successfully from: {selected_dir}")

    def on_select_wafer_clicked(self):
        """Handles the click event for select wafer button"""
        if not hasattr(self, 'current_user_dir') or not self.current_user_dir:
            QMessageBox.warning(self, "No User Selected", "Please select a user first.")
            return
        
        try:
            # Create custom dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Wafer Selection")
            dialog.setMinimumWidth(600)
            dialog.setMinimumHeight(400)
            
            # Remove question mark button and add resizing
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
            
            layout = QVBoxLayout(dialog)
            layout.setSpacing(20)
            layout.setContentsMargins(20, 20, 20, 20)
            
            # Title
            title = QLabel("Select Wafer")
            title.setFont(QFont("Arial", int(16 * FONT_SIZE_MULTIPLIER), QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)
            
            # Instructions
            instruction = QLabel("Choose an existing wafer or create a new one:")
            instruction.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
            layout.addWidget(instruction)
            
            # Get existing wafers
            existing_wafers = []
            if os.path.exists(self.current_user_dir):
                existing_wafers = [d for d in os.listdir(self.current_user_dir) 
                                if os.path.isdir(os.path.join(self.current_user_dir, d))]
            
            # Filter out only valid wafer directories
            valid_wafers = []
            for wafer in existing_wafers:
                wafer_dir = os.path.join(self.current_user_dir, wafer)
                # Consider it valid if it at least has one of the required directories
                required_dirs = [TRAINED_DIR, UNTRAINED_DIR, MODEL_DIR, 'dead', 'quick_measure']
                if any(os.path.exists(os.path.join(wafer_dir, d)) for d in required_dirs):
                    valid_wafers.append(wafer)
            
            # List of existing wafers
            wafer_list = QListWidget()
            wafer_list.setFont(QFont("Arial", BASE_FONT_SIZE))
            
            if valid_wafers:
                wafer_list.addItems(valid_wafers)
                # Select current wafer if it exists
                if self.current_wafer_name:
                    index = wafer_list.findText(self.current_wafer_name)
                    if index >= 0:
                        wafer_list.setCurrentRow(index)
            else:
                wafer_list.addItem("No existing wafers")
                wafer_list.setEnabled(False)
            
            layout.addWidget(wafer_list)
            
            # Buttons
            button_layout = QHBoxLayout()
            button_layout.setSpacing(15)
            
            select_btn = QPushButton("Select Wafer")
            select_btn.setEnabled(bool(valid_wafers))
            select_btn.setStyleSheet("background-color: #007bff; color: white;")
            select_btn.clicked.connect(dialog.accept)
            button_layout.addWidget(select_btn)
            
            new_wafer_btn = QPushButton("Create New Wafer")
            new_wafer_btn.setStyleSheet("background-color: #28a745; color: white;")
            new_wafer_btn.clicked.connect(lambda: dialog.done(2))  # Special return code
            button_layout.addWidget(new_wafer_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            
            # Add resize handler to ensure proper layout
            def resizeEvent(event):
                dialog.updateGeometry()
                
            dialog.resizeEvent = resizeEvent
            
            # Show dialog
            result = dialog.exec_()
            
            if result == QDialog.Accepted and valid_wafers:
                # User selected an existing wafer
                selected_wafer = wafer_list.currentItem().text()
                directory = os.path.join(self.current_user_dir, selected_wafer)
            elif result == 2:
                # User wants to create a new wafer
                directory = self.create_new_wafer_dialog()
                if not directory:
                    return  # User cancelled or error occurred
            else:
                return  # User cancelled
            
            # Check if directory has the expected structure, if not, create it
            required_folders = [TRAINED_DIR, UNTRAINED_DIR, MODEL_DIR, 'dead', 'quick_measure']
            for folder in required_folders:
                folder_path = os.path.join(directory, folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            
            # Store the wafer directory and name
            self.current_wafer_dir = directory
            self.current_wafer_name = os.path.basename(directory)
            
            # Update wafer labels
            self.wafer_label.setText(f"Wafer: {self.current_wafer_name}")
            if hasattr(self, 'baking_wafer_label'):
                self.baking_wafer_label.setText(self.current_wafer_name)
            
            # Update file counts
            self.update_file_counts()
            
            # Find and load the most recent model
            self.load_most_recent_model()
            
            self.status_label.setText(f"Wafer '{self.current_wafer_name}' selected.")
            
        except Exception as e:
            print(f"Error in on_select_wafer_clicked: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

    def create_new_wafer_dialog(self):
        """Show dialog to create a new wafer"""
        if not hasattr(self, 'current_user_dir') or not self.current_user_dir:
            QMessageBox.warning(self, "No User Selected", "Please select a user first.")
            return None
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Wafer")
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(250)
        
        # Remove question mark button and add resizing
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Instructions
        instruction = QLabel("Enter a name for the new wafer:")
        instruction.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        layout.addWidget(instruction)
        
        # Wafer name input
        name_input = QLineEdit()
        name_input.setFont(QFont("Arial", BASE_FONT_SIZE))
        layout.addWidget(name_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        create_btn = QPushButton("Create Wafer")
        create_btn.setStyleSheet("background-color: #28a745; color: white;")
        create_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(create_btn)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted and name_input.text().strip():
            wafer_name = name_input.text().strip()
            wafer_dir = os.path.join(self.current_user_dir, wafer_name)
            
            # Check if wafer already exists
            if os.path.exists(wafer_dir):
                QMessageBox.warning(self, "Wafer Exists", f"A wafer named '{wafer_name}' already exists.")
                return None
            
            # Create wafer directory and subdirectories
            os.makedirs(wafer_dir)
            os.makedirs(os.path.join(wafer_dir, TRAINED_DIR))
            os.makedirs(os.path.join(wafer_dir, UNTRAINED_DIR))
            os.makedirs(os.path.join(wafer_dir, MODEL_DIR))
            os.makedirs(os.path.join(wafer_dir, 'quick_measure'))
            
            return wafer_dir
        
        return None

    def show_model_selection(self):
        """Show dialog to select a model for the current wafer"""
        if not self.current_wafer_dir:
            QMessageBox.warning(self, "No Wafer Selected", "Please select a wafer first.")
            return
            
        # Get model directory
        model_dir = os.path.join(self.current_wafer_dir, MODEL_DIR)
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Get all model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if not model_files:
            QMessageBox.information(self, "No Models", f"No models found for wafer '{self.current_wafer_name}'.\n\nUse 'Retrain Model' to create a model.")
            return
            
        # Create model selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model")
        dialog.setMinimumWidth(1200)  # Increased width for additional columns
        dialog.setMinimumHeight(600)
        
        # Remove question mark button and add resizing
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setSpacing(20)
        dialog_layout.setContentsMargins(20, 20, 20, 20)
        
        # Add explanation at the top
        explanation_label = QLabel("Errors shown in percentage points (e.g., 2.5pp = 2.5% prediction error)")
        explanation_label.setFont(QFont("Arial", int(10 * FONT_SIZE_MULTIPLIER)))
        explanation_label.setStyleSheet("color: #6c757d; padding: 5px;")
        explanation_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(explanation_label)
        
        # Create a table to display model information
        model_table = QTableWidget()
        model_table.setColumnCount(7)  # Keep 7 columns
        model_table.setHorizontalHeaderLabels([
            "Model", "Created On", "Files Used", "Train RMSE (pp)", "Test RMSE (pp)", "Train MAE (pp)", "Test MAE (pp)"
        ])
        
        # Set column resize modes
        model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 7):
            model_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        # Populate table with model information
        model_table.setRowCount(len(model_files))
        
        # Sort model files by date (newest first)
        model_files.sort(reverse=True)
        
        for i, model_file in enumerate(model_files):
            # Extract timestamp from filename
            timestamp_str = model_file.replace(f"{self.current_wafer_name}_model_", "").replace(".pkl", "")
            
            try:
                # Convert timestamp to readable date
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                date_str = "Unknown"
            
            # Try to load model to get metadata
            try:
                model_path = os.path.join(model_dir, model_file)
                model = joblib.load(model_path)
                
                if hasattr(model, 'metadata'):
                    file_count = model.metadata.get('file_count', "Unknown")
                    
                    # Get training metrics (these are in percentage units, display as percentage points)
                    metrics = model.metadata.get('metrics', {})
                    train_rmse_pct = metrics.get('train_rmse', "N/A")
                    test_rmse_pct = metrics.get('test_rmse', "N/A")
                    train_mae_pct = metrics.get('train_mae', "N/A")
                    test_mae_pct = metrics.get('test_mae', "N/A")
                    
                    # Format as percentage points (just display the values with pp units)
                    train_rmse_str = f"{train_rmse_pct:.1f}" if isinstance(train_rmse_pct, (int, float)) else str(train_rmse_pct)
                    test_rmse_str = f"{test_rmse_pct:.1f}" if isinstance(test_rmse_pct, (int, float)) else str(test_rmse_pct)
                    train_mae_str = f"{train_mae_pct:.1f}" if isinstance(train_mae_pct, (int, float)) else str(train_mae_pct)
                    test_mae_str = f"{test_mae_pct:.1f}" if isinstance(test_mae_pct, (int, float)) else str(test_mae_pct)
                else:
                    file_count = "Unknown"
                    train_rmse_str = test_rmse_str = train_mae_str = test_mae_str = "N/A"
            except:
                file_count = "Unknown"
                train_rmse_str = test_rmse_str = train_mae_str = test_mae_str = "N/A"
            
            # Set table items
            model_table.setItem(i, 0, QTableWidgetItem(model_file))
            model_table.setItem(i, 1, QTableWidgetItem(date_str))
            model_table.setItem(i, 2, QTableWidgetItem(str(file_count)))
            model_table.setItem(i, 3, QTableWidgetItem(train_rmse_str))
            model_table.setItem(i, 4, QTableWidgetItem(test_rmse_str))
            model_table.setItem(i, 5, QTableWidgetItem(train_mae_str))
            model_table.setItem(i, 6, QTableWidgetItem(test_mae_str))
        
        dialog_layout.addWidget(model_table)
        
        # Add buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        select_btn = QPushButton("Select Model")
        select_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(select_btn)
        
        dialog_layout.addLayout(button_layout)
        
        # Add resize handler
        def resizeEvent(event):
            dialog.updateGeometry()
            model_table.updateGeometry()
            
        dialog.resizeEvent = resizeEvent
        
        # Show the dialog
        if dialog.exec_() == QDialog.Accepted and model_table.currentRow() >= 0:
            # Get selected model file
            selected_model = model_table.item(model_table.currentRow(), 0).text()
            model_path = os.path.join(model_dir, selected_model)
            
            # Load the model
            try:
                self.current_model_path = model_path
                self.current_model_label.setText(selected_model)
                self.status_label.setText(f"Selected model: {selected_model}")
                
                # Update model info in UI
                self.show_model_info()
                
                # If there's an active baking thread, update its model
                if self.baking_thread and hasattr(self.baking_thread, 'model_path'):
                    self.baking_thread.model_path = model_path
                    try:
                        self.baking_thread.model = joblib.load(model_path)
                    except:
                        pass
                
            except Exception as e:
                QMessageBox.critical(self, "Model Error", f"Error loading model: {str(e)}")

    def show_model_info(self):
        """Show detailed information about the current model"""
        if not self.current_model_path or not os.path.exists(self.current_model_path):
            QMessageBox.information(self, "No Model", "No model is currently selected.")
            return
            
        try:
            # Load the model to get metadata
            model = joblib.load(self.current_model_path)
            
            # Create info dialog
            info_dialog = QDialog(self)
            info_dialog.setWindowTitle("Model Information")
            info_dialog.setMinimumWidth(700)
            info_dialog.setMinimumHeight(500)
            
            # Remove question mark button and add resizing
            info_dialog.setWindowFlags(info_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
            
            info_layout = QVBoxLayout(info_dialog)
            info_layout.setSpacing(20)
            info_layout.setContentsMargins(20, 20, 20, 20)
            
            # Model filename
            title_label = QLabel(f"Model: {os.path.basename(self.current_model_path)}")
            title_label.setFont(QFont("Arial", int(16 * FONT_SIZE_MULTIPLIER), QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            info_layout.addWidget(title_label)
            
            # Extract metadata
            if hasattr(model, 'metadata'):
                metadata = model.metadata
                creation_date = metadata.get('creation_date', "Unknown")
                file_count = metadata.get('file_count', "Unknown")
                trained_files = metadata.get('trained_files', 0)
                untrained_files = metadata.get('untrained_files', 0)
                metrics = metadata.get('metrics', {})
                
                # Format creation date
                try:
                    creation_date = datetime.strptime(creation_date, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                except:
                    creation_date = "Unknown"
            else:
                # Extract info from filename
                model_filename = os.path.basename(self.current_model_path)
                timestamp_str = model_filename.replace(f"{self.current_wafer_name}_model_", "").replace(".pkl", "")
                
                try:
                    # Convert timestamp to readable date
                    creation_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
                except:
                    creation_date = "Unknown"
                    
                file_count = "Unknown"
                trained_files = "Unknown"
                untrained_files = "Unknown"
                metrics = {}
            
            # Create tabbed widget for organized information
            tab_widget = QTabWidget()
            
            # General Information Tab
            general_tab = QWidget()
            general_layout = QVBoxLayout(general_tab)
            
            general_layout.addWidget(QLabel(f"Created on: {creation_date}"))
            general_layout.addWidget(QLabel(f"Total files used: {file_count}"))
            general_layout.addWidget(QLabel(f"Previously trained files: {trained_files}"))
            general_layout.addWidget(QLabel(f"New files added: {untrained_files}"))
            general_layout.addWidget(QLabel(f"Model type: Predicts percentage change from last baking to cooled"))
            general_layout.addWidget(QLabel(f"Constraints: 1.5x to 2.3x magnitude of initial drop"))
            
            # Model Performance Tab
            performance_tab = QWidget()
            performance_layout = QVBoxLayout(performance_tab)
            
            # Add explanation at the top
            perf_explanation = QLabel("Prediction errors in percentage points (pp)\nExample: 2.5pp = 2.5% prediction error")
            perf_explanation.setFont(QFont("Arial", int(10 * FONT_SIZE_MULTIPLIER)))
            perf_explanation.setStyleSheet("color: #6c757d; padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
            perf_explanation.setAlignment(Qt.AlignCenter)
            performance_layout.addWidget(perf_explanation)
            
            # Create performance table
            if metrics:
                perf_frame = QFrame()
                perf_frame.setFrameShape(QFrame.Box)
                perf_frame.setStyleSheet("background-color: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px; padding: 15px;")
                perf_layout = QVBoxLayout(perf_frame)
                
                perf_title = QLabel("Model Performance Metrics")
                perf_title.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
                perf_title.setAlignment(Qt.AlignCenter)
                perf_layout.addWidget(perf_title)
                
                # Training metrics (display as percentage points)
                train_rmse_pct = metrics.get('train_rmse', 'N/A')
                train_mae_pct = metrics.get('train_mae', 'N/A')
                test_rmse_pct = metrics.get('test_rmse', 'N/A')
                test_mae_pct = metrics.get('test_mae', 'N/A')
                
                # Format as percentage points
                train_rmse_str = f"{train_rmse_pct:.1f}pp" if isinstance(train_rmse_pct, (int, float)) else str(train_rmse_pct)
                test_rmse_str = f"{test_rmse_pct:.1f}pp" if isinstance(test_rmse_pct, (int, float)) else str(test_rmse_pct)
                train_mae_str = f"{train_mae_pct:.1f}pp" if isinstance(train_mae_pct, (int, float)) else str(train_mae_pct)
                test_mae_str = f"{test_mae_pct:.1f}pp" if isinstance(test_mae_pct, (int, float)) else str(test_mae_pct)
                
                metrics_layout = QGridLayout()
                
                # Headers
                metrics_layout.addWidget(QLabel("Metric"), 0, 0)
                metrics_layout.addWidget(QLabel("Training"), 0, 1)
                metrics_layout.addWidget(QLabel("Testing"), 0, 2)
                
                # RMSE row
                rmse_label = QLabel("RMSE")
                rmse_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
                metrics_layout.addWidget(rmse_label, 1, 0)
                
                train_rmse_label = QLabel(train_rmse_str)
                train_rmse_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
                train_rmse_label.setStyleSheet("color: #007bff;")
                metrics_layout.addWidget(train_rmse_label, 1, 1)
                
                test_rmse_label = QLabel(test_rmse_str)
                test_rmse_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
                test_rmse_label.setStyleSheet("color: #28a745;")
                metrics_layout.addWidget(test_rmse_label, 1, 2)
                
                # MAE row
                mae_label = QLabel("MAE")
                mae_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER), QFont.Bold))
                metrics_layout.addWidget(mae_label, 2, 0)
                
                train_mae_label = QLabel(train_mae_str)
                train_mae_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
                train_mae_label.setStyleSheet("color: #007bff;")
                metrics_layout.addWidget(train_mae_label, 2, 1)
                
                test_mae_label = QLabel(test_mae_str)
                test_mae_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
                test_mae_label.setStyleSheet("color: #28a745;")
                metrics_layout.addWidget(test_mae_label, 2, 2)
                
                perf_layout.addLayout(metrics_layout)
                performance_layout.addWidget(perf_frame)
                
                # Add explanation
                explanation = QLabel("RMSE: Root Mean Square Error (lower is better)\nMAE: Mean Absolute Error (lower is better)")
                explanation.setFont(QFont("Arial", int(10 * FONT_SIZE_MULTIPLIER)))
                explanation.setStyleSheet("color: #6c757d;")
                performance_layout.addWidget(explanation)
            else:
                performance_layout.addWidget(QLabel("No performance metrics available"))
            
            # Add tabs to widget (only General and Performance)
            tab_widget.addTab(general_tab, "General")
            tab_widget.addTab(performance_tab, "Performance")
            
            info_layout.addWidget(tab_widget)
            
            # OK button
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            ok_btn = QPushButton("OK")
            ok_btn.clicked.connect(info_dialog.accept)
            button_layout.addWidget(ok_btn)
            
            info_layout.addLayout(button_layout)
            
            # Show the dialog
            info_dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Model Info Error", f"Could not get model information: {str(e)}")
            
    def update_untrained_count(self):
        """Update the count of untrained files"""
        if not self.current_wafer_dir:
            self.untrained_count_label.setText("0")
            self.retrain_model_btn.setEnabled(False)
            return
            
        untrained_dir = os.path.join(self.current_wafer_dir, UNTRAINED_DIR)
        if not os.path.exists(untrained_dir):
            self.untrained_count_label.setText("0")
            # self.retrain_model_btn.setEnabled(False)
            return
            
        untrained_files = [f for f in os.listdir(untrained_dir) if f.endswith('.csv')]
        count = len(untrained_files)
        
        self.untrained_count_label.setText(str(count))
        
        # Enable retrain button if there are untrained files
        self.retrain_model_btn.setEnabled(count >= 0)

    def load_most_recent_model(self):
        """Find and load the most recent model for the current wafer"""
        if not self.current_wafer_dir:
            self.current_model_path = None
            self.current_model_label.setText("None Selected")
            return
            
        model_dir = os.path.join(self.current_wafer_dir, MODEL_DIR)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            self.current_model_path = None
            self.current_model_label.setText("No Models Available")
            return
            
        # Get all model files and sort by name (which includes timestamp)
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if not model_files:
            self.current_model_path = None
            self.current_model_label.setText("No Models Available")
            return
            
        # Sort by name (newest first)
        model_files.sort(reverse=True)
        
        # Load the most recent model
        most_recent_model = model_files[0]
        self.current_model_path = os.path.join(model_dir, most_recent_model)
        self.current_model_label.setText(most_recent_model)
        
        # Try to get model creation date
        try:
            # Extract timestamp from filename
            timestamp_str = most_recent_model.replace(f"{self.current_wafer_name}_model_", "").replace(".pkl", "")
            creation_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")
        except:
            creation_date = "Unknown"
        
        # Show message about latest model with option to change
        if len(model_files) > 1:
            reply = QMessageBox.question(
                self, 
                "Latest Model Selected", 
                f"Latest model has been automatically selected:\n\n"
                f"Model: {most_recent_model}\n"
                f"Created: {creation_date}\n"
                f"Total models available: {len(model_files)}\n\n"
                f"Would you like to select a different model?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.show_model_selection()
        else:
            # Only one model available
            QMessageBox.information(
                self, 
                "Model Selected", 
                f"Model automatically selected:\n\n"
                f"Model: {most_recent_model}\n"
                f"Created: {creation_date}\n"
                f"This is the only model available for this wafer."
            )
        
        # Try to get trained file count from model metadata
        try:
            model = joblib.load(self.current_model_path)
            if hasattr(model, 'metadata'):
                file_count = model.metadata.get('file_count', 0)
                # Update trained files count label
                if hasattr(self, 'trained_count_label'):
                    self.trained_count_label.setText(str(file_count))
        except Exception as e:
            print(f"Error loading model metadata: {e}")

    def toggle_measurement(self):
        """Start or stop the quick measurement process"""
        print("Toggle measurement called")
        try:
            if hasattr(self, 'measurement_thread') and self.measurement_thread and self.measurement_thread.running:
                # Stop measurement
                self.measurement_thread.stop()
                self.measurement_thread.wait()  # Wait for thread to finish
                self.quick_measure_btn.setText("Quick Measurement")
                self.baking_save_btn.setEnabled(True)  # Enable save after measurement
                
                # Update status - using status_label instead of baking_status_text
                if hasattr(self, 'status_label'):
                    self.status_label.setText("Quick measurement completed")
                
                # Update the main window's status bar as fallback
                self.statusBar().showMessage("Quick measurement completed")
            else:
                # Validate wafer selection
                if not self.current_wafer_dir:
                    QMessageBox.warning(self, "No Wafer Selected", "Please select a wafer before starting.")
                    return
                
                # Start measurement
                self.quick_measure_btn.setText("Stop Measuring")
                
                # Clear existing plot data
                if hasattr(self, 'baking_resistance_curve'):
                    self.baking_resistance_curve.setData([], [])
                if hasattr(self, 'voltage_curve'):
                    self.voltage_curve.setData([], [])
                
                # Remove any prediction point and error bars
                if hasattr(self, 'predicted_resistance_point'):
                    self.predicted_resistance_point.setData([], [])
                    
                if hasattr(self, 'error_bar_item') and self.error_bar_item is not None:
                    if isinstance(self.error_bar_item, list):
                        for item in self.error_bar_item:
                            self.baking_plot.removeItem(item)
                    else:
                        self.baking_plot.removeItem(self.error_bar_item)
                    self.error_bar_item = None
                
                # Reset measurement display
                if hasattr(self, 'current_resistance_label'):
                    self.current_resistance_label.setText("--- Ω")
                if hasattr(self, 'predicted_resistance_label'):
                    self.predicted_resistance_label.setText("--- Ω")
                
                # Create and start measurement thread
                self.measurement_thread = MeasurementThread()
                self.measurement_thread.measurement_update.connect(self.update_measurement)
                self.measurement_thread.measurement_complete.connect(self.on_measurement_complete)
                self.measurement_thread.error_occurred.connect(self.on_measurement_error)
                self.measurement_thread.start()
                
                # Disable save button during measurement
                if hasattr(self, 'baking_save_btn'):
                    self.baking_save_btn.setEnabled(False)
                
                # Update status - using status_label instead of baking_status_text
                if hasattr(self, 'status_label'):
                    self.status_label.setText("Quick measurement in progress...")
                
                # Update the main window's status bar as fallback
                self.statusBar().showMessage("Quick measurement in progress...")
        except Exception as e:
            print(f"Error in toggle_measurement: {e}")
            import traceback
            traceback.print_exc()

    def update_measurement(self, time_val, resistance):
        """Update UI with new measurement data from quick measurement"""
        try:
            # Update the plot with scatter points
            if hasattr(self, 'measurement_thread'):
                time_data, resistance_data, _ = self.measurement_thread.get_data()
                
                # Update scatter plot with new points
                if hasattr(self, 'baking_resistance_curve'):
                    self.baking_resistance_curve.setData(
                        x=time_data, y=resistance_data, 
                        size=10, brush=pg.mkBrush(255, 0, 0)
                    )
                
                # Adjust the plot view to show all data with some margin
                if len(time_data) > 1 and hasattr(self, 'baking_plot'):
                    self.baking_plot.setXRange(0, max(time_data) * 1.1)
                    min_r = min(resistance_data) * 0.95
                    max_r = max(resistance_data) * 1.05
                    self.baking_plot.setYRange(min_r, max_r)
                
                # Update the measurement display
                if hasattr(self, 'current_resistance_label'):
                    self.current_resistance_label.setText(f"{resistance:.2f} Ω")
                
                # Update change display
                self.update_change_display(resistance)
                
                # Store the last resistance
                self.last_resistance = resistance
                
        except Exception as e:
            print(f"Error in update_measurement: {e}")

    def initial_measurement(self):
        """First step of baking workflow: Measure initial resistance"""
        # Create dialog with proper sizing and no question mark
        dialog = QDialog(self)
        dialog.setWindowTitle("Initial Resistance Measurement")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)
        
        # Remove question mark button and add proper window controls
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Initial Resistance Measurement")
        title.setFont(QFont("Arial", int(16 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0275d8;")
        layout.addWidget(title)
        
        # Simple instruction
        instruction_label = QLabel("Measure resistance before placing chip on hot plate")
        instruction_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label)
        
        # Resistance display frame
        resistance_frame = QFrame()
        resistance_frame.setFrameShape(QFrame.Box)
        resistance_frame.setStyleSheet("background-color: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px; padding: 15px;")
        resistance_layout = QVBoxLayout(resistance_frame)
        resistance_layout.setSpacing(15)
        
        reading_title = QLabel("Current Reading")
        reading_title.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        reading_title.setAlignment(Qt.AlignCenter)
        resistance_layout.addWidget(reading_title)
        
        self.initial_r_label = QLabel("--- Ω")
        self.initial_r_label.setFont(QFont("Arial", int(24 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.initial_r_label.setAlignment(Qt.AlignCenter)
        self.initial_r_label.setStyleSheet("color: #d9534f;")
        resistance_layout.addWidget(self.initial_r_label)
        
        layout.addWidget(resistance_frame)
        
        # Status message
        self.initial_status_label = QLabel("Ready to measure")
        self.initial_status_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        self.initial_status_label.setAlignment(Qt.AlignCenter)
        self.initial_status_label.setStyleSheet("color: #5a6268;")
        layout.addWidget(self.initial_status_label)
        
        # Create plot with better sizing
        self.initial_plot = pg.PlotWidget()
        self.initial_plot.setBackground('w')
        self.initial_plot.setMinimumHeight(250)
        self.initial_plot.setLabel('left', 'Resistance', units='Ω')
        self.initial_plot.setLabel('bottom', 'Time', units='s')
        
        label_style = {'font-size': f'{int(9 * FONT_SIZE_MULTIPLIER)}pt'}
        axis_style = {'tickFont': QFont('Arial', int(7 * FONT_SIZE_MULTIPLIER))}
        self.initial_plot.getAxis('left').setLabel('Resistance', units='Ω', **label_style)
        self.initial_plot.getAxis('bottom').setLabel('Time', units='s', **label_style)
        self.initial_plot.getAxis('left').setStyle(**axis_style)
        self.initial_plot.getAxis('bottom').setStyle(**axis_style)
        self.initial_plot.showGrid(x=True, y=True)
        
        self.initial_resistance_scatter = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(255, 0, 0), pen=None, name='Resistance'
        )
        self.initial_plot.addItem(self.initial_resistance_scatter)
        layout.addWidget(self.initial_plot)
        
        # Button layout with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # Toggle Start/Stop button
        self.start_stop_btn = QPushButton("Start Measurement")
        self.start_stop_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.start_stop_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
        self.start_stop_btn.clicked.connect(self.toggle_initial_measurement)
        button_layout.addWidget(self.start_stop_btn)
        
        # Add Clear Data button
        clear_btn = QPushButton("Clear Data")
        clear_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        clear_btn.setStyleSheet("background-color: #ffc107; color: black; border-radius: 5px; padding: 10px 20px;")
        clear_btn.clicked.connect(self.clear_initial_data)
        button_layout.addWidget(clear_btn)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        cancel_btn.setStyleSheet("background-color: #6c757d; color: white; border-radius: 5px; padding: 10px 20px;")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Store dialog reference
        self.initial_measurement_dialog = dialog
        
        # Initialize variables
        self.initial_measurement_thread = None
        self.initial_r_values = []
        self.is_measuring = False
        
        # Show dialog
        dialog.exec_()

    def toggle_initial_measurement(self):
        """Toggle between start and stop for initial measurement"""
        if not self.is_measuring:
            # Start measurement
            self.start_initial_measurement()
        else:
            # Stop measurement
            self.stop_initial_measurement()

    def start_initial_measurement(self):
        """Start measuring initial resistance"""
        try:
            self.initial_r_values = []
            self.is_measuring = True
            
            # Update button to Stop with red color
            self.start_stop_btn.setText("Stop Measurement")
            self.start_stop_btn.setStyleSheet("background-color: #dc3545; color: white; border-radius: 5px; padding: 10px 20px;")
            
            # Create thread for initial measurement
            self.initial_measurement_thread = MeasurementThread()
            
            # Connect signals
            self.initial_measurement_thread.measurement_update.connect(self.update_initial_measurement)
            self.initial_measurement_thread.measurement_complete.connect(self.on_initial_measurement_complete)
            self.initial_measurement_thread.error_occurred.connect(self.on_initial_measurement_error)
            
            # Start thread
            self.initial_measurement_thread.start()
            
            # Update status
            self.initial_status_label.setText("Measuring in progress")
            
        except Exception as e:
            print(f"Error in start_initial_measurement: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to start measurement: {str(e)}")
            self.is_measuring = False
            self.start_stop_btn.setText("Start Measurement")
            self.start_stop_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")

    def update_initial_measurement(self, time_val, resistance):
        """Update UI during initial resistance measurement"""
        try:
            # Store the resistance value
            self.initial_r_values.append(resistance)
            
            # Update the label with current value
            self.initial_r_label.setText(f"{resistance:.2f} Ω")
            
            # Update the plot
            if hasattr(self, 'initial_measurement_thread'):
                time_data, resistance_data, _ = self.initial_measurement_thread.get_data()
                self.initial_resistance_scatter.setData(x=time_data, y=resistance_data)
                
                # Adjust plot range
                if len(time_data) > 1:
                    self.initial_plot.setXRange(0, max(time_data) * 1.1)
                    min_r = min(resistance_data) * 0.95
                    max_r = max(resistance_data) * 1.05
                    self.initial_plot.setYRange(min_r, max_r)
            
            # Update status without showing reading count
            self.initial_status_label.setText("Measuring in progress")
            
        except Exception as e:
            print(f"Error in update_initial_measurement: {e}")


    def clear_initial_data(self):
        """Stop measurement and clear initial measurement data"""
        # Stop any ongoing initial measurement
        if hasattr(self, 'initial_measurement_thread') and self.initial_measurement_thread and self.initial_measurement_thread.running:
            self.initial_measurement_thread.stop()
            self.initial_measurement_thread.wait()
        
        # Clear data
        self.initial_r_values = []
        if hasattr(self, 'initial_measurement_thread') and self.initial_measurement_thread:
            self.initial_measurement_thread.time_data = []
            self.initial_measurement_thread.resistance_data = []
            self.initial_measurement_thread.voltage_data = []
        
        # Clear plot with empty data arrays
        if hasattr(self, 'initial_resistance_scatter'):
            self.initial_resistance_scatter.setData([], [])
        
        # Reset displays
        if hasattr(self, 'initial_r_label'):
            self.initial_r_label.setText("--- Ω")
        if hasattr(self, 'initial_status_label'):
            self.initial_status_label.setText("Data cleared - Ready to measure")
        
        # Reset button state
        if hasattr(self, 'start_stop_btn'):
            self.start_stop_btn.setText("Start Measurement")
            self.start_stop_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
            self.is_measuring = False
    def stop_initial_measurement(self):
        """Stop measuring initial resistance and calculate average"""
        if self.initial_measurement_thread and self.initial_measurement_thread.running:
            self.initial_measurement_thread.stop()
            self.initial_measurement_thread.wait()
        
        self.is_measuring = False
        
        # Update button back to Start with green color
        self.start_stop_btn.setText("Start Measurement")
        self.start_stop_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
        
        # Calculate average resistance
        if self.initial_r_values:
            self.initial_r_avg = sum(self.initial_r_values) / len(self.initial_r_values)
            self.initial_r_label.setText(f"{self.initial_r_avg:.2f} Ω (Average)")
            self.initial_status_label.setText(f"Initial resistance measured: {self.initial_r_avg:.2f} Ω")
            
            # AUTO-CLOSE the dialog and proceed to next step
            self.initial_measurement_dialog.accept()
            
            # Show the "move to hot plate" message and start baking
            self.show_move_to_hotplate_and_start_baking(self.initial_r_avg)
        else:
            self.initial_status_label.setText("No measurements recorded. Try again.")

    def on_initial_measurement_complete(self):
        """Handle completion of initial measurement"""
        try:
            self.is_measuring = False
            # Don't automatically calculate average here - let user stop manually
        except Exception as e:
            print(f"Error in on_initial_measurement_complete: {e}")

    def on_initial_measurement_error(self, error_msg):
        """Handle initial measurement errors"""
        try:
            self.is_measuring = False
            self.start_stop_btn.setText("Start Measurement")
            self.start_stop_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
            self.initial_status_label.setText(f"Error: {error_msg}")
            QMessageBox.critical(self, "Measurement Error", f"An error occurred: {error_msg}")
        except Exception as e:
            print(f"Error in on_initial_measurement_error: {e}")
    # Modify the show_move_to_hotplate_and_start_baking method (around line ~2460)
    def show_move_to_hotplate_and_start_baking(self, initial_resistance):
        """Show move to hotplate message, then start baking when user clicks Start Baking"""
        # Display move-to-hotplate dialog with proper sizing and no question mark
        hotplate_dialog = QDialog(self)
        hotplate_dialog.setWindowTitle("Move to Hot Plate")
        hotplate_dialog.setMinimumWidth(700)
        hotplate_dialog.setMinimumHeight(400)
        
        # Remove question mark button and add proper window controls
        hotplate_dialog.setWindowFlags(hotplate_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        layout = QVBoxLayout(hotplate_dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Simple clear instruction in a prominent box
        instruction_frame = QFrame()
        instruction_frame.setFrameShape(QFrame.Box)
        instruction_frame.setStyleSheet("background-color: #f8d7da; border: 2px solid #f5c6cb; border-radius: 10px; padding: 15px;")
        instruction_layout = QVBoxLayout(instruction_frame)
        
        instruction_label = QLabel("MOVE CHIP TO HOT PLATE NOW")
        instruction_label.setFont(QFont("Arial", int(20 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: #721c24;")
        instruction_layout.addWidget(instruction_label)
        
        layout.addWidget(instruction_frame)
        
        # Information frame - modified to include target setting
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Box)
        info_frame.setStyleSheet("background-color: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px; padding: 15px;")
        info_layout = QGridLayout(info_frame)
        info_layout.setVerticalSpacing(15)
        info_layout.setHorizontalSpacing(15)
        
        # Display measured initial resistance
        info_layout.addWidget(QLabel("Initial Resistance:"), 0, 0)
        r_value_label = QLabel(f"{initial_resistance:.2f} Ω")
        r_value_label.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        r_value_label.setStyleSheet("color: #d9534f;")
        info_layout.addWidget(r_value_label, 0, 1)
        
        # Target resistance with input to change - NEW ADDITION
        info_layout.addWidget(QLabel("Target Resistance:"), 1, 0)
        
        # Create a horizontal layout for the target input
        target_layout = QHBoxLayout()
        
        # Get current target value from the main screen
        current_target = self.target_resistance_input.value()
        
        # Create target input field
        self.hotplate_target_input = QDoubleSpinBox()
        self.hotplate_target_input.setMinimum(0)
        self.hotplate_target_input.setMaximum(100000)
        self.hotplate_target_input.setValue(current_target)
        self.hotplate_target_input.setFont(QFont("Arial", BASE_FONT_SIZE, QFont.Bold))
        self.hotplate_target_input.setStyleSheet("color: #28a745;")
        self.hotplate_target_input.setSingleStep(100)  # More convenient step size
        target_layout.addWidget(self.hotplate_target_input)
        
        # Add Ω unit label
        target_unit = QLabel("Ω")
        target_unit.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        target_unit.setStyleSheet("color: #28a745;")
        target_layout.addWidget(target_unit)
        
        info_layout.addLayout(target_layout, 1, 1)
        
        # Auto-stop checkbox - NEW ADDITION
        self.hotplate_auto_stop = QCheckBox("Auto-stop at target")
        self.hotplate_auto_stop.setFont(QFont("Arial", BASE_FONT_SIZE))
        self.hotplate_auto_stop.setChecked(True)  # Set checked by default
        info_layout.addWidget(self.hotplate_auto_stop, 2, 0, 1, 2)
        
        layout.addWidget(info_frame)
        
        # Buttons with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        back_btn = QPushButton("Back")
        back_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        back_btn.setStyleSheet("background-color: #6c757d; color: white; border-radius: 5px; padding: 10px 20px;")
        back_btn.clicked.connect(hotplate_dialog.reject)
        button_layout.addWidget(back_btn)
        
        start_btn = QPushButton("Start Baking")
        start_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        start_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
        start_btn.clicked.connect(hotplate_dialog.accept)
        button_layout.addWidget(start_btn)
        
        layout.addLayout(button_layout)
        
        # Show the dialog
        result = hotplate_dialog.exec_()
        
        # If user clicked Start Baking, begin the baking process on main screen
        if result == QDialog.Accepted:
            # Update the main target input with the new value from dialog
            new_target = self.hotplate_target_input.value()
            self.target_resistance_input.setValue(new_target)
            
            # Update the auto-stop setting
            auto_stop_setting = self.hotplate_auto_stop.isChecked()
            self.auto_stop_checkbox.setChecked(auto_stop_setting)
            
            # Start baking with updated settings
            self._start_baking_process(initial_resistance)        
    def get_current_limit_value(self):
        """Convert current limit dropdown selection to float value in Amperes"""
        current_limit_text = self.current_limit_dropdown.currentText()
        value = float(current_limit_text.split()[0])  # Extract number
        if "µA" in current_limit_text:
            return value * 1e-6  # Convert µA to A
        elif "mA" in current_limit_text:
            return value * 1e-3  # Convert mA to A
        else:
            return value  # Assume it's already in A

    def _start_baking_process(self, initial_resistance):
        """Start the actual baking measurement with stored initial resistance"""
        # Update target line
        target_resistance = self.target_resistance_input.value()
        self.target_line.setValue(target_resistance)
        
        # Clear existing plot data
        self.baking_resistance_curve.setData([], [])
        self.voltage_curve.setData([], [])
        
        # Remove any prediction point and error bars
        self.predicted_resistance_point.setData([], [])
        if self.error_bar_item is not None:
            if isinstance(self.error_bar_item, list):
                for item in self.error_bar_item:
                    self.baking_plot.removeItem(item)
            else:
                self.baking_plot.removeItem(self.error_bar_item)
            self.error_bar_item = None
        
        # Update button
        self.baking_start_btn.setText("Stop Baking")
        
        # Show initial resistance value
        self.current_resistance_label.setText(f"{initial_resistance:.2f} Ω")
        self.predicted_resistance_label.setText("--- Ω")
        
        # Store initial resistance for later use
        self.initial_resistance = initial_resistance
        
        # Get UI parameters
        voltage = self.voltage_input.value()
        frequency = self.frequency_input.value() 
        current_limit = self.get_current_limit_value()
        
        print(f"Creating baking thread with: V={voltage}V, f={frequency}Hz, I_limit={current_limit*1e6}µA")
        
        # Create and start baking thread with UI parameters
        self.baking_thread = BakingThread(
            model_path=self.current_model_path,
            voltage=voltage,
            frequency=frequency,
            current_limit=current_limit
        )
        self.baking_thread.temperature = self.baking_temp_input.value()
        self.baking_thread.initial_resistance = initial_resistance
        
        # Set target and offset
        if self.auto_stop_checkbox.isChecked():
            offset_ohms = self.offset_input.value()
            self.baking_thread.set_target(
                target_resistance, 
                auto_stop=True,
                offset_ohms=offset_ohms
            )
        
        # Connect signals
        self.baking_thread.measurement_update.connect(self.update_baking_measurement)
        self.baking_thread.prediction_update.connect(self.update_baking_prediction)
        self.baking_thread.measurement_complete.connect(self.on_baking_complete)
        self.baking_thread.error_occurred.connect(self.on_baking_error)
        
        # Start thread
        self.baking_thread.start()
        
        # Disable save button during baking
        self.baking_save_btn.setEnabled(False)
        
        # Update status - using status_label instead of baking_status_text
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Baking in progress... Initial R: {initial_resistance:.2f} Ω")
        
        # Update the main window's status bar as fallback
        self.statusBar().showMessage(f"Baking in progress... Initial R: {initial_resistance:.2f} Ω")

    def update_baking_measurement(self, time_val, resistance, voltage):
        """Update baking UI with new measurement data"""
        # Skip NaN resistance values for display but keep them in the data
        if np.isnan(resistance):
            return
            
        # Update the plots
        time_data, resistance_data, voltage_data = self.baking_thread.get_data()
        
        # Filter out NaN values for plotting resistance
        valid_indices = [i for i, r in enumerate(resistance_data) if not np.isnan(r)]
        valid_times = [time_data[i] for i in valid_indices]
        valid_resistances = [resistance_data[i] for i in valid_indices]
        
        # Update scatter plot with valid points
        if valid_resistances:
            self.baking_resistance_curve.setData(
                x=valid_times, y=valid_resistances, 
                size=10, brush=pg.mkBrush(255, 0, 0)
            )
            
            # Store the last valid resistance
            self.last_resistance = valid_resistances[-1]
            
            # Update the measurement display
            self.current_resistance_label.setText(f"{self.last_resistance:.2f} Ω")
            
            # Update change display
            self.update_change_display(self.last_resistance)
        
        # Update voltage scatter plot
        self.voltage_curve.setData(
            x=time_data, y=voltage_data, 
            size=8, brush=pg.mkBrush(0, 0, 255)
        )
        
        # Adjust the resistance plot view
        if valid_times:
            self.baking_plot.setXRange(0, max(valid_times) * 1.1)
            min_r = min(valid_resistances) * 0.95
            max_r = max(valid_resistances) * 1.05
            self.baking_plot.setYRange(min_r, max_r)
        
        # Adjust the voltage plot view
        if time_data:
            self.voltage_plot.setXRange(0, max(time_data) * 1.1)
            if voltage_data:
                min_v = min(voltage_data) * 1.2  # Give more room for negative values
                max_v = max(voltage_data) * 1.2
                self.voltage_plot.setYRange(min_v, max_v)

    ### update_baking_prediction ###
    def update_baking_prediction(self, predicted_r):
        """Update the predicted resistance display"""
        # Store prediction
        self.last_prediction = predicted_r

        # Update widget display
        self.predicted_resistance_label.setText(f"{predicted_r:.2f} Ω")
    
    def on_baking_complete(self):
        """Handle baking completion with automatic cooled measurement dialog"""
        """Handle baking completion with automatic cooled measurement dialog"""
        self.baking_start_btn.setText("Start Baking")
        self.baking_save_btn.setEnabled(True)
        
        # Update status - using status_label instead of baking_status_text
        if hasattr(self, 'status_label'):
            self.status_label.setText("Baking process completed")
        
        # Update the main window's status bar as fallback
        self.statusBar().showMessage("Baking process completed")
        
        # Store baking end timestamp and final resistance from thread
        if self.baking_thread:
            self.baking_end_timestamp = self.baking_thread.baking_end_timestamp
            self.final_baking_resistance = self.baking_thread.final_baking_resistance
        
        # Show the "move to cool plate" dialog with warning about timing
        self.show_cooled_measurement_dialog()
    def on_baking_error(self, error_msg):
        """Handle baking errors"""
        QMessageBox.critical(self, "Baking Error", f"An error occurred: {error_msg}")
        self.baking_start_btn.setText("Start Baking")
        
        # Update status - using status_label instead of baking_status_text
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Error: {error_msg}")
        
        # Update the main window's status bar as fallback
        self.statusBar().showMessage(f"Error: {error_msg}")
    def show_cooled_measurement_dialog(self):
        """Show dialog asking user to move to cool plate and measure final resistance"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Final Cooled Measurement")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)
        
        # Remove question mark button and add proper window controls
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Baking Complete")
        title.setFont(QFont("Arial", int(16 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #28a745;")
        layout.addWidget(title)
        
        # Instructions frame - simplified
        instruction_frame = QFrame()
        instruction_frame.setFrameShape(QFrame.Box)
        instruction_frame.setStyleSheet("background-color: #f8d7da; border: 2px solid #f5c6cb; border-radius: 10px; padding: 15px;")
        instruction_layout = QVBoxLayout(instruction_frame)
        
        instruction_label = QLabel("MOVE CHIP TO COOL PLATE")
        instruction_label.setFont(QFont("Arial", int(18 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("color: #721c24;")
        instruction_layout.addWidget(instruction_label)
        
        wait_label = QLabel("Wait 10 seconds for chip to cool down")
        wait_label.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER)))
        wait_label.setAlignment(Qt.AlignCenter)
        wait_label.setStyleSheet("color: #721c24;")
        instruction_layout.addWidget(wait_label)
        
        layout.addWidget(instruction_frame)
        
        # Warning frame about timing - minimal text
        warning_frame = QFrame()
        warning_frame.setFrameShape(QFrame.Box)
        warning_frame.setStyleSheet("background-color: #fff3cd; border: 2px solid #ffeaa7; border-radius: 10px; padding: 15px;")
        warning_layout = QVBoxLayout(warning_frame)
        
        warning_title = QLabel("⚠️ IMPORTANT")
        warning_title.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        warning_title.setAlignment(Qt.AlignCenter)
        warning_title.setStyleSheet("color: #856404;")
        warning_layout.addWidget(warning_title)
        
        warning_text = QLabel("Start cooled measurement within 10 minutes of baking completion.")
        warning_text.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        warning_text.setAlignment(Qt.AlignCenter)
        warning_text.setWordWrap(True)
        warning_text.setStyleSheet("color: #856404;")
        warning_layout.addWidget(warning_text)
        
        layout.addWidget(warning_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        skip_btn = QPushButton("Skip (chip is dead)")
        skip_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        skip_btn.setStyleSheet("background-color: #6c757d; color: white; border-radius: 5px; padding: 10px 20px;")
        skip_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(skip_btn)
        
        measure_btn = QPushButton("Start Cooled Measurement")
        measure_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        measure_btn.setStyleSheet("background-color: #007bff; color: white; border-radius: 5px; padding: 10px 20px;")
        measure_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(measure_btn)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            self.start_cooled_measurement()

    def start_cooled_measurement(self):
        """Start the cooled measurement process"""
        # Store cooled start timestamp
        self.cooled_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create dialog with proper sizing and no question mark
        dialog = QDialog(self)
        dialog.setWindowTitle("Cooled Resistance Measurement")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)
        
        # Remove question mark button and add proper window controls
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title 
        title = QLabel("Cooled Resistance Measurement")
        title.setFont(QFont("Arial", int(16 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #17a2b8;")
        layout.addWidget(title)
        
        # Brief instruction
        instruction_label = QLabel("Measure the resistance after cooling")
        instruction_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label)
        
        # Resistance display frame
        resistance_frame = QFrame()
        resistance_frame.setFrameShape(QFrame.Box)
        resistance_frame.setStyleSheet("background-color: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px; padding: 15px;")
        resistance_layout = QVBoxLayout(resistance_frame)
        resistance_layout.setSpacing(15)
        
        reading_title = QLabel("Current Reading")
        reading_title.setFont(QFont("Arial", int(14 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        reading_title.setAlignment(Qt.AlignCenter)
        resistance_layout.addWidget(reading_title)
        
        self.cooled_r_label = QLabel("--- Ω")
        self.cooled_r_label.setFont(QFont("Arial", int(24 * FONT_SIZE_MULTIPLIER), QFont.Bold))
        self.cooled_r_label.setAlignment(Qt.AlignCenter)
        self.cooled_r_label.setStyleSheet("color: #17a2b8;")
        resistance_layout.addWidget(self.cooled_r_label)
        
        layout.addWidget(resistance_frame)
        
        # Status message
        self.cooled_status_label = QLabel("Ready to measure")
        self.cooled_status_label.setFont(QFont("Arial", int(12 * FONT_SIZE_MULTIPLIER)))
        self.cooled_status_label.setAlignment(Qt.AlignCenter)
        self.cooled_status_label.setStyleSheet("color: #5a6268;")
        layout.addWidget(self.cooled_status_label)
        
        # Create plot with better sizing
        self.cooled_plot = pg.PlotWidget()
        self.cooled_plot.setBackground('w')
        self.cooled_plot.setMinimumHeight(250)
        self.cooled_plot.setLabel('left', 'Resistance', units='Ω')
        self.cooled_plot.setLabel('bottom', 'Time', units='s')
        
        label_style = {'font-size': f'{int(9 * FONT_SIZE_MULTIPLIER)}pt'}
        axis_style = {'tickFont': QFont('Arial', int(7 * FONT_SIZE_MULTIPLIER))}
        self.cooled_plot.getAxis('left').setLabel('Resistance', units='Ω', **label_style)
        self.cooled_plot.getAxis('bottom').setLabel('Time', units='s', **label_style)
        self.cooled_plot.getAxis('left').setStyle(**axis_style)
        self.cooled_plot.getAxis('bottom').setStyle(**axis_style)
        self.cooled_plot.showGrid(x=True, y=True)
        
        self.cooled_resistance_scatter = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush(23, 162, 184), pen=None, name='Cooled Resistance'
        )
        self.cooled_plot.addItem(self.cooled_resistance_scatter)
        layout.addWidget(self.cooled_plot)
        
        # Buttons with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        
        # Single Start/Stop toggle button
        self.cooled_toggle_btn = QPushButton("Start Measurement")
        self.cooled_toggle_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        self.cooled_toggle_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
        self.cooled_toggle_btn.clicked.connect(self.toggle_cooled_measurement)
        button_layout.addWidget(self.cooled_toggle_btn)
        
        # Clear Data button
        clear_cooled_btn = QPushButton("Clear Data")
        clear_cooled_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        clear_cooled_btn.setStyleSheet("background-color: #ffc107; color: black; border-radius: 5px; padding: 10px 20px;")
        clear_cooled_btn.clicked.connect(self.clear_cooled_data)
        button_layout.addWidget(clear_cooled_btn)
        
        # End measurement button
        end_measurement_btn = QPushButton("End Measurement")
        end_measurement_btn.setFont(QFont("Arial", BUTTON_FONT_SIZE))
        end_measurement_btn.setStyleSheet("background-color: #007bff; color: white; border-radius: 5px; padding: 10px 20px;")
        end_measurement_btn.clicked.connect(lambda: self.end_cooled_measurement(dialog))
        self.end_measurement_btn = end_measurement_btn
        button_layout.addWidget(end_measurement_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize variables
        self.cooled_measurement_thread = None
        self.cooled_r_values = []
        self.cooled_is_measuring = False
        self.cooled_measurement_dialog = dialog
        
        # Show dialog
        dialog.exec_()

    def toggle_cooled_measurement(self):
        """Toggle between start and stop for cooled measurement"""
        if not self.cooled_is_measuring:
            # Start measurement
            self.start_cooled_measuring()
        else:
            # Stop measurement
            self.stop_cooled_measuring()

    def start_cooled_measuring(self):
        """Start measuring cooled resistance"""
        self.cooled_r_values = []
        self.cooled_is_measuring = True
        
        # Update toggle button to show Stop
        self.cooled_toggle_btn.setText("Stop Measurement")
        self.cooled_toggle_btn.setStyleSheet("background-color: #dc3545; color: white; border-radius: 5px; padding: 10px 20px;")
        
        # Create thread for cooled measurement
        self.cooled_measurement_thread = MeasurementThread()
        
        # Connect signals
        self.cooled_measurement_thread.measurement_update.connect(self.update_cooled_measurement)
        
        # Start thread
        self.cooled_measurement_thread.start()
        
        # Update status
        self.cooled_status_label.setText("Measuring cooled resistance...")

    def stop_cooled_measuring(self):
        """Stop measuring cooled resistance and calculate average"""
        if self.cooled_measurement_thread and self.cooled_measurement_thread.running:
            self.cooled_measurement_thread.stop()
            self.cooled_measurement_thread.wait()
        
        self.cooled_is_measuring = False
        
        # Update toggle button to show Start
        self.cooled_toggle_btn.setText("Start Measurement")
        self.cooled_toggle_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
        
        # Calculate average resistance
        if self.cooled_r_values:
            self.cooled_r_avg = sum(self.cooled_r_values) / len(self.cooled_r_values)
            self.cooled_r_label.setText(f"{self.cooled_r_avg:.2f} Ω (Average)")
            self.cooled_status_label.setText(f"Cooled resistance measured: {self.cooled_r_avg:.2f} Ω")
        else:
            self.cooled_status_label.setText("No measurements recorded. Try again.")

    def update_cooled_measurement(self, time_val, resistance):
        """Update display with new cooled resistance measurement"""
        # Store resistance value
        self.cooled_r_values.append(resistance)
        
        # Update label with current value
        self.cooled_r_label.setText(f"{resistance:.2f} Ω")
        
        # Update plot
        time_data, resistance_data, _ = self.cooled_measurement_thread.get_data()
        self.cooled_resistance_scatter.setData(x=time_data, y=resistance_data)
        
        # Adjust plot range
        if len(time_data) > 1:
            self.cooled_plot.setXRange(0, max(time_data) * 1.1)
            min_r = min(resistance_data) * 0.95
            max_r = max(resistance_data) * 1.05
            self.cooled_plot.setYRange(min_r, max_r)

    def clear_cooled_data(self):
        """Stop measurement and clear cooled measurement data"""
        # Stop any ongoing cooled measurement
        if hasattr(self, 'cooled_measurement_thread') and self.cooled_measurement_thread and self.cooled_measurement_thread.running:
            self.cooled_measurement_thread.stop()
            self.cooled_measurement_thread.wait()
        
        # Clear cooled data
        self.cooled_r_values = []
        if hasattr(self, 'cooled_measurement_thread') and self.cooled_measurement_thread:
            self.cooled_measurement_thread.time_data = []
            self.cooled_measurement_thread.resistance_data = []
            self.cooled_measurement_thread.voltage_data = []
        
        # Clear plot with empty data arrays
        if hasattr(self, 'cooled_resistance_scatter'):
            self.cooled_resistance_scatter.setData([], [])
        
        # Reset displays
        if hasattr(self, 'cooled_r_label'):
            self.cooled_r_label.setText("--- Ω")
        if hasattr(self, 'cooled_status_label'):
            self.cooled_status_label.setText("Data cleared - Ready to measure")
        
        # Reset button state
        if hasattr(self, 'cooled_toggle_btn'):
            self.cooled_toggle_btn.setText("Start Measurement")
            self.cooled_toggle_btn.setStyleSheet("background-color: #28a745; color: white; border-radius: 5px; padding: 10px 20px;")
            self.cooled_is_measuring = False
    def end_cooled_measurement(self, dialog):
        """End cooled measurement, calculate average, save unified and separate files"""
        # Stop measurement if running
        if self.cooled_measurement_thread and self.cooled_measurement_thread.running:
            self.cooled_measurement_thread.stop()
            self.cooled_measurement_thread.wait()
        
        # Calculate average resistance
        if not self.cooled_r_values:
            QMessageBox.warning(self, "No Data", "No cooled measurement data to save.")
            return
        
        self.cooled_r_avg = sum(self.cooled_r_values) / len(self.cooled_r_values)
        
        # Check for dead chip (80% drop from final baking resistance)
        dead_chip = False
        dead_chip_reason = ""
        if self.final_baking_resistance and self.cooled_r_avg:
            resistance_drop = (self.final_baking_resistance - self.cooled_r_avg) / self.final_baking_resistance
            if resistance_drop >= 0.8:  # 80% drop
                dead_chip = True
                dead_chip_reason = f"Dead chip detected: {resistance_drop*100:.1f}% drop in resistance"
        
        # Check timing between baking end and cooled start
        timing_valid = True
        timing_reason = ""
        if self.baking_end_timestamp and self.cooled_start_timestamp:
            try:
                baking_end = datetime.strptime(self.baking_end_timestamp, "%Y-%m-%d %H:%M:%S")
                cooled_start = datetime.strptime(self.cooled_start_timestamp, "%Y-%m-%d %H:%M:%S")
                time_diff = (cooled_start - baking_end).total_seconds() / 60  # minutes
                
                if time_diff > 10:
                    timing_valid = False
                    timing_reason = f"Too much time between baking and cooled: {time_diff:.1f} minutes"
            except Exception as e:
                print(f"Error checking timing: {e}")
        
        # Determine save location based on chip status and timing
        if dead_chip:
            save_dir = os.path.join(self.current_wafer_dir, "dead")
            save_status = "Dead Chip"
        elif not timing_valid:
            save_dir = os.path.join(self.current_wafer_dir, "dead") 
            save_status = "Invalid Timing"
        else:
            save_dir = os.path.join(self.current_wafer_dir, UNTRAINED_DIR)
            save_status = "Valid Training Data"
        
        # Get chip information
        wafer_name = self.current_wafer_name
        row = self.baking_row_input.value()
        chip = self.baking_chip_input.text().strip()
        temp = self.baking_temp_input.value()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save separate raw cooled data file
            cooled_raw_dir = os.path.join(self.current_wafer_dir, "cooled_raw")
            if not os.path.exists(cooled_raw_dir):
                os.makedirs(cooled_raw_dir)
            
            cooled_filename = f"{wafer_name}_R{row}_C{chip}_{temp}C_cooled_raw_{timestamp}.csv"
            cooled_file_path = os.path.join(cooled_raw_dir, cooled_filename)
            
            # Get cooled data
            cooled_time_data, cooled_resistance_data, cooled_voltage_data = self.cooled_measurement_thread.get_data()
            
            # Create cooled DataFrame with timestamps
            cooled_data = {
                'Time (s)': cooled_time_data,
                'Resistance (Ohms)': cooled_resistance_data,
                'Voltage (V)': cooled_voltage_data,
                'Timestamp': [self.cooled_start_timestamp] * len(cooled_time_data)
            }
            
            cooled_df = pd.DataFrame(cooled_data)
            cooled_df.to_csv(cooled_file_path, index=False)
            
            # Save unified file with baking data and summary averages
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            unified_filename = f"{wafer_name}_R{row}_C{chip}_{temp}C_unified_{timestamp}.csv"
            unified_file_path = os.path.join(save_dir, unified_filename)
            
            # Get baking data
            if self.baking_thread:
                baking_time_data, baking_resistance_data, baking_voltage_data = self.baking_thread.get_data()
            else:
                # Fallback if no baking data
                baking_time_data = [0]
                baking_resistance_data = [0]
                baking_voltage_data = [0]
            
            # Create unified DataFrame with correct format
            unified_data = {
                'Time (s)': baking_time_data,
                'Resistance (Ohms)': baking_resistance_data,
                'Voltage (V)': baking_voltage_data,
            }
            
            # Add summary values only in first row
            n_rows = len(baking_time_data)
            unified_data['Cooled_Average_Resistance'] = [None] * n_rows
            unified_data['Initial_Resistance (Ohms)'] = [None] * n_rows
            unified_data['Baking_End_Timestamp'] = [None] * n_rows
            unified_data['Cooled_Start_Timestamp'] = [None] * n_rows
            unified_data['Dead_Chip'] = [None] * n_rows
            unified_data['Timing_Valid'] = [None] * n_rows
            unified_data['Notes'] = [None] * n_rows
            
            # Set values only for first row
            if n_rows > 0:
                unified_data['Cooled_Average_Resistance'][0] = self.cooled_r_avg
                if hasattr(self, 'initial_resistance') and self.initial_resistance:
                    unified_data['Initial_Resistance (Ohms)'][0] = self.initial_resistance
                unified_data['Baking_End_Timestamp'][0] = self.baking_end_timestamp
                unified_data['Cooled_Start_Timestamp'][0] = self.cooled_start_timestamp
                unified_data['Dead_Chip'][0] = dead_chip
                unified_data['Timing_Valid'][0] = timing_valid
                unified_data['Notes'][0] = dead_chip_reason or timing_reason or ""
            
            unified_df = pd.DataFrame(unified_data)
            unified_df.to_csv(unified_file_path, index=False)
            
            # Update file counts
            self.update_file_counts()
            
            # Show completion message
            message = f"Measurement complete!\n\n"
            message += f"Status: {save_status}\n"
            message += f"Cooled Average: {self.cooled_r_avg:.2f} Ω\n"
            if dead_chip:
                message += f"⚠️ {dead_chip_reason}\n"
            if not timing_valid:
                message += f"⚠️ {timing_reason}\n"
            message += f"\nFiles saved:\n"
            message += f"• Raw cooled: {cooled_filename}\n"
            message += f"• Unified: {unified_filename}\n"
            message += f"• Location: {os.path.basename(save_dir)}"
            
            QMessageBox.information(self, "Measurement Complete", message)
            
            # Close the dialog
            dialog.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
            print(f"Error in end_cooled_measurement: {e}")
            import traceback
            traceback.print_exc()

    def clear_baking_data(self):
        """Stop any ongoing process and clear all measurement and baking data"""
        # Stop any ongoing process
        if hasattr(self, 'measurement_thread') and self.measurement_thread and self.measurement_thread.running:
            self.measurement_thread.stop()
            self.measurement_thread.wait()
            self.quick_measure_btn.setText("🔬 Quick Measurement")
        
        if hasattr(self, 'baking_thread') and self.baking_thread and self.baking_thread.running:
            self.baking_thread.stop()
            self.baking_thread.wait()
            self.baking_start_btn.setText("🔥 Start Baking")
        
        # Clear measurement data
        if hasattr(self, 'measurement_thread') and self.measurement_thread:
            self.measurement_thread.time_data = []
            self.measurement_thread.resistance_data = []
            self.measurement_thread.voltage_data = []
        
        # Clear baking data
        if hasattr(self, 'baking_thread') and self.baking_thread:
            self.baking_thread.time_data = []
            self.baking_thread.resistance_data = []
            self.baking_thread.voltage_data = []
        
        # Clear plots with empty data arrays to properly clear visible points
        if hasattr(self, 'baking_resistance_curve'):
            self.baking_resistance_curve.setData([], [])
        
        if hasattr(self, 'voltage_curve'):
            self.voltage_curve.setData([], [])
        
        # Remove prediction point (blue dot) and error bars completely
        if hasattr(self, 'predicted_resistance_point'):
            self.predicted_resistance_point.setData([], [])
        
        if hasattr(self, 'error_bar_item') and self.error_bar_item is not None:
            if isinstance(self.error_bar_item, list):
                for item in self.error_bar_item:
                    self.baking_plot.removeItem(item)
            else:
                self.baking_plot.removeItem(self.error_bar_item)
            self.error_bar_item = None
            
        # Reset measurement display
        if hasattr(self, 'current_resistance_label'):
            self.current_resistance_label.setText("--- Ω")
        
        if hasattr(self, 'predicted_resistance_label'):
            self.predicted_resistance_label.setText("--- Ω")
        
        
        # Reset change display
        if hasattr(self, 'change_display_label'):
            self.change_display_label.setText("--- Ω (---%)")
            
        # Reset predictions and stored values
        self.last_resistance = None
        self.last_prediction = None
        self.initial_resistance = None
        
        if hasattr(self, 'baking_save_btn'):
            self.baking_save_btn.setEnabled(False)
    
    def update_change_display(self, current_resistance):
        """Update the change display with current resistance change from initial"""
        if hasattr(self, 'initial_resistance') and self.initial_resistance is not None and current_resistance is not None:
            change_ohms = current_resistance - self.initial_resistance
            if self.initial_resistance != 0:
                change_percent = 100 * change_ohms / self.initial_resistance
            else:
                change_percent = 0
            
            # Format the display with + or - sign
            sign = "+" if change_ohms >= 0 else ""
            self.change_display_label.setText(f"{sign}{change_ohms:.2f} Ω ({change_percent:+.1f}%)")
            
            # Color coding: green for positive change, red for negative
            if change_ohms >= 0:
                self.change_display_label.setStyleSheet("color: #28a745; margin: 5px 0;")  # Green
            else:
                self.change_display_label.setStyleSheet("color: #dc3545; margin: 5px 0;")  # Red
        else:
            self.change_display_label.setText("--- Ω (---%)")
            self.change_display_label.setStyleSheet("color: #0275d8; margin: 5px 0;")  # Default blue
    def toggle_baking(self):
        """Start or stop the baking process with improved user workflow"""
        if self.baking_thread and self.baking_thread.running:
            # Stop baking
            self.baking_thread.stop()
            self.baking_thread.wait()  # Wait for thread to finish
            self.baking_start_btn.setText("🔥 Start Baking")
            self.baking_save_btn.setEnabled(True)
            
            # Update status - using status_label instead of baking_status_text
            if hasattr(self, 'status_label'):
                self.status_label.setText("Baking process completed")
            
            # Update the main window's status bar as fallback
            self.statusBar().showMessage("Baking process completed")
        else:
            # Validate input
            if not self.current_wafer_dir:
                QMessageBox.warning(self, "No Wafer Selected", "Please select a wafer before starting.")
                return
                
            if not self.baking_chip_input.text().strip():
                QMessageBox.warning(self, "Input Required", "Please enter a chip identifier before starting.")
                return
            
            # Improved workflow - Step 1: Measure initial resistance first
            self.initial_measurement()
    def save_baking_data(self):
        """Save data to CSV file with improved workflow logic"""
        if not self.current_wafer_dir or not self.current_wafer_name:
            QMessageBox.warning(self, "No Wafer Selected", "Please select a wafer before saving.")
            return
        
        # Determine data source and type
        if self.measurement_thread and len(self.measurement_thread.time_data) > 0:
            # Save quick measurement data
            time_data, resistance_data, voltage_data = self.measurement_thread.get_data()
            data_type = "Quick Measurement"
            save_dir = os.path.join(self.current_wafer_dir, "quick_measure")
        elif self.baking_thread and len(self.baking_thread.time_data) > 0:
            # Save baking data  
            time_data, resistance_data, voltage_data = self.baking_thread.get_data()
            data_type = "Baking"
            save_dir = os.path.join(self.current_wafer_dir, UNTRAINED_DIR)
        else:
            QMessageBox.warning(self, "No Data", "There is no data to save.")
            return
        
        # Get selected wafer name
        wafer_name = self.current_wafer_name
        
        # Get chip information
        row = self.baking_row_input.value()
        chip = self.baking_chip_input.text().strip()
        temp = self.baking_temp_input.value()
        
        # Create filename based on type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if data_type == "Quick Measurement":
            filename = f"{wafer_name}_R{row}_C{chip}_{temp}C_quick_{timestamp}.csv"
        else:  # Baking
            filename = f"{wafer_name}_R{row}_C{chip}_{temp}C_baking_{timestamp}.csv"
        
        # Ensure directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create full file path
        file_path = os.path.join(save_dir, filename)
        
        try:
            # Create DataFrame with the data
            data = {
                'Time (s)': time_data,
                'Resistance (Ohms)': resistance_data,
                'Voltage (V)': voltage_data
            }
            
            # Add initial resistance if available (only in first row)
            if hasattr(self, 'initial_resistance') and self.initial_resistance is not None:
                data['Initial_Resistance (Ohms)'] = [None] * len(time_data)
                if len(time_data) > 0:
                    data['Initial_Resistance (Ohms)'][0] = self.initial_resistance
            else:
                # Fallback: use the first valid resistance
                valid_resistances = [r for r in resistance_data if not np.isnan(r)]
                if valid_resistances:
                    initial_r = valid_resistances[0]
                    data['Initial_Resistance (Ohms)'] = [None] * len(time_data)
                    if len(time_data) > 0:
                        data['Initial_Resistance (Ohms)'][0] = initial_r
            
            # Add prediction data if it was a baking process (only in first row)
            if data_type == "Baking" and hasattr(self, 'last_prediction') and self.last_prediction is not None:
                data['Predicted_Final_Resistance'] = [None] * len(time_data)
                if len(time_data) > 0:
                    data['Predicted_Final_Resistance'][0] = self.last_prediction
                
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            # Update file counts
            self.update_file_counts()
            
            folder_name = "quick_measure" if data_type == "Quick Measurement" else "untrained"
            QMessageBox.information(self, "Save Successful", 
                                f"Data saved to {folder_name} folder as:\n{filename}")
            
            # Update status - using status_label instead of baking_status_text
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Data saved to {folder_name} folder")
            
            # Update the main window's status bar as fallback
            self.statusBar().showMessage(f"Data saved to {folder_name} folder")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
            
            # Update status - using status_label instead of baking_status_text
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Error saving data: {str(e)}")
            
            # Update the main window's status bar as fallback
            self.statusBar().showMessage(f"Error saving data: {str(e)}")

    def on_measurement_complete(self):
        """Handle quick measurement completion"""
        try:
            if hasattr(self, 'quick_measure_btn'):
                self.quick_measure_btn.setText("Quick Measurement")
            if hasattr(self, 'baking_save_btn'):
                self.baking_save_btn.setEnabled(True)
            if hasattr(self, 'baking_status_text'):
                self.baking_status_text.setText("Quick measurement completed")
                self.baking_status_text.setStyleSheet("color: green;")
        except Exception as e:
            print(f"Error in on_measurement_complete: {e}")

    def on_measurement_error(self, error_msg):
        """Handle measurement errors"""
        try:
            QMessageBox.critical(self, "Measurement Error", f"An error occurred: {error_msg}")
            if hasattr(self, 'quick_measure_btn'):
                self.quick_measure_btn.setText("Quick Measurement")
            if hasattr(self, 'baking_status_text'):
                self.baking_status_text.setText(f"Error: {error_msg}")
                self.baking_status_text.setStyleSheet("color: red;")
        except Exception as e:
            print(f"Error in on_measurement_error: {e}")

# Main entry point
def main():
        app = QApplication(sys.argv)
        # Set default font size for the entire application
        font = QFont("Arial", BASE_FONT_SIZE)
        app.setFont(font)
    
        # Exception hook to catch and display unhandled exceptions
        def exception_hook(exctype, value, traceback):
            print(exctype, value, traceback)
            sys.__excepthook__(exctype, value, traceback)
            sys.exit(1)
    
        sys.excepthook = exception_hook
    
        # Create and show the main window

        window = ABAAController()
        window.show()
    
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
