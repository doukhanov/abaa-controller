import os
import glob
import re
import shutil
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Directory structure constants
TRAINED_DIR = "trained"  # Subdirectory for files the model was trained on
UNTRAINED_DIR = "untrained"  # Subdirectory for files the model was not trained on
PARTIAL_DIR = "partial"  # Subdirectory for partial measurements
DEAD_DIR = "dead"  # Subdirectory for dead chips
MODEL_DIR = "models"  # Directory for model files

# Dead chip detection thresholds
DEAD_CHIP_DROP_THRESHOLD = 0.5  # 50% resistance drop
COOLING_TIME_THRESHOLD = 10  # 10 minutes max between baking end and cooling start

class PreprocessingThread:
    """Thread for preprocessing data files before model training"""
    
    def __init__(self, wafer_dir):
        """Initialize with wafer directory path"""
        self.wafer_dir = wafer_dir
        
    def run(self):
        """Run the preprocessing steps for the wafer directory"""
        try:
            # Setup directories
            self.progress_update(5, "Setting up directories...")
            untrained_dir = os.path.join(self.wafer_dir, UNTRAINED_DIR)
            trained_dir = os.path.join(self.wafer_dir, TRAINED_DIR)
            partial_dir = os.path.join(self.wafer_dir, PARTIAL_DIR)
            dead_dir = os.path.join(self.wafer_dir, DEAD_DIR)
            
            # Create directories if they don't exist
            for directory in [untrained_dir, trained_dir, partial_dir, dead_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            # Identify files in untrained directory
            self.progress_update(10, "Identifying files...")
            
            # Get all CSV files from untrained directory
            all_files = []
            if os.path.exists(untrained_dir):
                all_files.extend(glob.glob(os.path.join(untrained_dir, "*.csv")))
            
            self.progress_update(20, f"Found {len(all_files)} files for processing")
            
            # Process each file
            for idx, file_path in enumerate(all_files):
                progress = 20 + (70 * (idx + 1) / max(len(all_files), 1))
                self.progress_update(int(progress), f"Processing file {idx+1} of {len(all_files)}...")
                
                try:
                    self.process_unified_file(file_path, untrained_dir, trained_dir, partial_dir, dead_dir)
                except Exception as e:
                    self.progress_update(int(progress), f"Error processing file {os.path.basename(file_path)}: {str(e)}")
            
            # Complete
            self.progress_update(100, "Preprocessing complete")
            self.preprocessing_complete()
            
        except Exception as e:
            self.error_occurred(f"Preprocessing error: {str(e)}")
    
    def process_unified_file(self, file_path, untrained_dir, trained_dir, partial_dir, dead_dir):
        """Process a unified file that contains both baking and cooled data"""
        file_name = os.path.basename(file_path)
        self.progress_update(50, f"Processing file: {file_name}")
        
        # Determine if this is a unified file (look for 'unified' in filename)
        if 'unified' in file_name.lower():
            # It's a unified file, read it
            try:
                df = pd.read_csv(file_path)
                
                # Check if it has the necessary columns
                required_columns = ['Time (s)', 'Resistance (Ohms)', 'Voltage (V)']
                if not all(col in df.columns for col in required_columns):
                    self.progress_update(55, f"Missing required columns in {file_name}, moving to partial")
                    dest_path = os.path.join(partial_dir, file_name)
                    shutil.move(file_path, dest_path)
                    return
                
                # Check if we have the metadata columns with timing info
                metadata_complete = True
                metadata_columns = [
                    'Cooled_Average_Resistance', 
                    'Initial_Resistance (Ohms)', 
                    'Baking_End_Timestamp', 
                    'Cooled_Start_Timestamp',
                    'Dead_Chip',
                    'Timing_Valid'
                ]
                
                for col in metadata_columns:
                    if col not in df.columns:
                        metadata_complete = False
                        self.progress_update(56, f"Missing metadata column: {col} in {file_name}")
                
                # If metadata is incomplete, we'll do our own analysis
                if not metadata_complete:
                    self.analyze_unified_file_without_metadata(df, file_path, untrained_dir, trained_dir, partial_dir, dead_dir)
                    return
                
                # Check if chip is already marked as dead in metadata
                dead_chip_flag = False
                if 'Dead_Chip' in df.columns:
                    # Get the first non-null value
                    dead_chip_values = df['Dead_Chip'].dropna()
                    if len(dead_chip_values) > 0 and dead_chip_values.iloc[0]:
                        dead_chip_flag = True
                        self.progress_update(60, f"File {file_name} marked as dead chip in metadata")
                
                # Check if timing is already marked as invalid in metadata
                timing_valid = True
                if 'Timing_Valid' in df.columns:
                    # Get the first non-null value
                    timing_values = df['Timing_Valid'].dropna()
                    if len(timing_values) > 0 and not timing_values.iloc[0]:
                        timing_valid = False
                        self.progress_update(65, f"File {file_name} marked with invalid timing in metadata")
                
                # If metadata indicates dead chip or invalid timing, move to dead directory
                if dead_chip_flag or not timing_valid:
                    dest_path = os.path.join(dead_dir, file_name)
                    shutil.move(file_path, dest_path)
                    self.progress_update(70, f"Moved file {file_name} to dead directory based on metadata")
                    return
                
                # If we got here, the file is valid for training
                # However, we should still verify the cooling timing and resistance drop
                self.verify_cooling_and_resistance(df, file_path, untrained_dir, trained_dir, dead_dir)
                
            except Exception as e:
                self.progress_update(75, f"Error analyzing file {file_name}: {str(e)}")
                # On error, move to partial directory
                dest_path = os.path.join(partial_dir, file_name)
                shutil.move(file_path, dest_path)
                
        else:
            # Legacy format or other format - use a different parser or move to partial
            self.handle_legacy_file(file_path, untrained_dir, trained_dir, partial_dir, dead_dir)
    
    def verify_cooling_and_resistance(self, df, file_path, untrained_dir, trained_dir, dead_dir):
        """Verify cooling timing and check for resistance drop even if metadata says it's valid"""
        file_name = os.path.basename(file_path)
        
        try:
            # Get timestamps from metadata if available
            baking_end_timestamp = None
            cooling_start_timestamp = None
            
            # Find the first non-null value
            if 'Baking_End_Timestamp' in df.columns:
                baking_end_values = df['Baking_End_Timestamp'].dropna()
                if len(baking_end_values) > 0:
                    baking_end_timestamp = baking_end_values.iloc[0]
            
            if 'Cooled_Start_Timestamp' in df.columns:
                cooling_start_values = df['Cooled_Start_Timestamp'].dropna()
                if len(cooling_start_values) > 0:
                    cooling_start_timestamp = cooling_start_values.iloc[0]
            
            # Check cooling timing if we have both timestamps
            if baking_end_timestamp and cooling_start_timestamp:
                try:
                    baking_end = datetime.strptime(str(baking_end_timestamp), "%Y-%m-%d %H:%M:%S")
                    cooling_start = datetime.strptime(str(cooling_start_timestamp), "%Y-%m-%d %H:%M:%S")
                    time_diff_minutes = (cooling_start - baking_end).total_seconds() / 60
                    
                    if time_diff_minutes > COOLING_TIME_THRESHOLD:
                        self.progress_update(80, f"Cooling started too late: {time_diff_minutes:.1f} minutes after baking")
                        dest_path = os.path.join(dead_dir, file_name)
                        shutil.move(file_path, dest_path)
                        return
                except Exception as e:
                    self.progress_update(81, f"Error parsing timestamps: {str(e)}")
            
            # Check for resistance drop in the cooled measurement
            if 'Cooled_Average_Resistance' in df.columns:
                cooled_avg_values = df['Cooled_Average_Resistance'].dropna()
                if len(cooled_avg_values) > 0:
                    cooled_avg = cooled_avg_values.iloc[0]
                    
                    # Get final baking resistance (average of last 2 seconds)
                    final_baking_r = self.get_final_baking_resistance(df)
                    
                    if final_baking_r is not None and cooled_avg is not None:
                        resistance_drop = (final_baking_r - cooled_avg) / final_baking_r
                        if resistance_drop >= DEAD_CHIP_DROP_THRESHOLD:
                            self.progress_update(85, f"Large resistance drop detected: {resistance_drop*100:.1f}%")
                            dest_path = os.path.join(dead_dir, file_name)
                            shutil.move(file_path, dest_path)
                            return
            
            # If we got here, the file is valid for training
            # Keep it in untrained directory
            self.progress_update(90, f"File {file_name} validated for training")
            
        except Exception as e:
            self.progress_update(95, f"Error verifying cooling and resistance: {str(e)}")
            # On error, keep the file in untrained directory
    
    def get_final_baking_resistance(self, df):
        """Get the average resistance from the last 2 seconds of baking data"""
        try:
            # Convert to numeric to be safe
            df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
            df['Resistance (Ohms)'] = pd.to_numeric(df['Resistance (Ohms)'], errors='coerce')
            
            # Remove NaN values
            df_clean = df.dropna(subset=['Time (s)', 'Resistance (Ohms)'])
            
            if len(df_clean) == 0:
                return None
            
            # Get the maximum time
            max_time = df_clean['Time (s)'].max()
            
            # Get measurements from the last 2 seconds
            last_2_seconds = df_clean[df_clean['Time (s)'] >= (max_time - 2)]
            
            if len(last_2_seconds) == 0:
                return None
            
            # Calculate average resistance in this window
            final_avg_resistance = last_2_seconds['Resistance (Ohms)'].mean()
            return final_avg_resistance
            
        except Exception as e:
            print(f"Error getting final baking resistance: {str(e)}")
            return None
    
    def analyze_unified_file_without_metadata(self, df, file_path, untrained_dir, trained_dir, partial_dir, dead_dir):
        """Analyze a unified file that doesn't have complete metadata"""
        file_name = os.path.basename(file_path)
        
        try:
            # Convert columns to numeric
            df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
            df['Resistance (Ohms)'] = pd.to_numeric(df['Resistance (Ohms)'], errors='coerce')
            
            # Clean up NaN values
            df_clean = df.dropna(subset=['Time (s)', 'Resistance (Ohms)'])
            
            if len(df_clean) < 10:  # Not enough data points
                self.progress_update(60, f"Not enough valid data points in {file_name}")
                dest_path = os.path.join(partial_dir, file_name)
                shutil.move(file_path, dest_path)
                return
            
            # Check for significant resistance drops (50% for at least 3 seconds)
            significant_drop = False
            
            # Get the initial resistance (or the max value in the first 20% of the data)
            n_initial = max(1, int(len(df_clean) * 0.2))
            initial_r = df_clean['Resistance (Ohms)'].iloc[:n_initial].max()
            
            # Get the final resistance (average of last 10% of data points)
            n_final = max(1, int(len(df_clean) * 0.1))
            final_r = df_clean['Resistance (Ohms)'].iloc[-n_final:].mean()
            
            # Calculate resistance drop
            if initial_r > 0:
                resistance_drop = (initial_r - final_r) / initial_r
                if resistance_drop >= DEAD_CHIP_DROP_THRESHOLD:
                    significant_drop = True
                    self.progress_update(65, f"Significant resistance drop detected: {resistance_drop*100:.1f}%")
            
            if significant_drop:
                dest_path = os.path.join(dead_dir, file_name)
                shutil.move(file_path, dest_path)
                self.progress_update(70, f"Moved file {file_name} to dead directory due to resistance drop")
                return
            
            # If no significant drop, keep in untrained directory
            self.progress_update(75, f"File {file_name} validated for training")
            
        except Exception as e:
            self.progress_update(80, f"Error analyzing file without metadata: {str(e)}")
            # On error, move to partial directory
            dest_path = os.path.join(partial_dir, file_name)
            shutil.move(file_path, dest_path)
    
    def handle_legacy_file(self, file_path, untrained_dir, trained_dir, partial_dir, dead_dir):
        """Handle legacy file formats"""
        file_name = os.path.basename(file_path)
        
        # Different patterns for different file types
        baking_pattern = re.compile(
            r'^(?P<wafer>.+?)_(?P<row>R\d+)_(?P<chip>C\d+)_.*_(?P<stage>baking)_(?P<timestamp>\d+)\.csv$'
        )
        
        cooled_pattern = re.compile(
            r'^(?P<wafer>.+?)_(?P<row>R\d+)_(?P<chip>C\d+)_.*_(?P<stage>baking)_(?P<timestamp>\d+)\.cooled\.csv$'
        )
        
        legacy_pattern = re.compile(
            r'^(?P<wafer>.+?)_(?P<row>R\d+)_(?P<chip>C\d+)_.*_(?P<stage>.+)_output\.csv$'
        )
        
        # Try to match patterns
        baking_match = baking_pattern.match(file_name)
        cooled_match = cooled_pattern.match(file_name)
        legacy_match = legacy_pattern.match(file_name)
        
        if baking_match or cooled_match or legacy_match:
            # This is a recognized legacy format, but we can't process it properly
            # Move to partial directory for manual review
            self.progress_update(85, f"Legacy format detected: {file_name}, moving to partial directory")
            dest_path = os.path.join(partial_dir, file_name)
            shutil.move(file_path, dest_path)
        else:
            # Unrecognized format, also move to partial
            self.progress_update(90, f"Unrecognized format: {file_name}, moving to partial directory")
            dest_path = os.path.join(partial_dir, file_name)
            shutil.move(file_path, dest_path)

    # Placeholder functions for progress updates and signals
    def progress_update(self, percentage, message):
        print(f"Progress: {percentage}% - {message}")
    
    def preprocessing_complete(self):
        print("Preprocessing complete!")
    
    def error_occurred(self, message):
        print(f"Error: {message}")


# For testing the preprocessing functionality directly
def main():
    # Get wafer directory from command line arg or use default
    import sys
    if len(sys.argv) > 1:
        wafer_dir = sys.argv[1]
    else:
        # Default test directory
        wafer_dir = "test_wafer"
    
    print(f"Processing wafer directory: {wafer_dir}")
    
    # Create and run preprocessor
    preprocessor = PreprocessingThread(wafer_dir)
    preprocessor.run()

if __name__ == "__main__":
    main()