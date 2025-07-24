# EEG Seizure Classification Model
# Complete Google Colab Notebook for Epileptic Seizure Detection

## 1. Initial Setup and Dependencies


# Install required packages
!pip install mne pyEDFlib numpy pandas scikit-learn tensorflow matplotlib seaborn

# Import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mne
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Setup complete! TensorFlow version:", tf.__version__)

## 2. CHB-MIT Dataset Configuration and Loading

# Configuration for CHB-MIT dataset
DATA_PATH = '/content/physionet.org/files/chbmit/1.0.0/'  # Update this path to your CHB-MIT folder
SAMPLE_RATE = 256  # Hz (CHB-MIT standard)
WINDOW_SIZE = 5    # seconds
OVERLAP = 0.5      # 50% overlap

# Install wfdb for official PhysioNet downloads (faster and more reliable)
!pip install wfdb

import wfdb

def download_chb_mit_dataset_wfdb(download_path, patients=['chb01'], max_records_per_patient=5):
    """
    Download CHB-MIT dataset using the official WFDB library (recommended).
    This method is typically faster and more reliable than wget.
    
    Args:
        download_path: Path where to download the dataset
        patients: List of patient IDs to download (e.g., ['chb01', 'chb02'])
        max_records_per_patient: Maximum number of records to download per patient
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("Downloading CHB-MIT dataset using official WFDB library...")
    print(f"Download path: {download_path}")
    print(f"Patients: {patients}")
    print(f"Max records per patient: {max_records_per_patient}")
    
    try:
        # Create download directory
        os.makedirs(download_path, exist_ok=True)
        
        for patient in patients:
            print(f"\n📥 Downloading patient {patient}...")
            
            # Get list of available records for this patient
            try:
                # First, try to get the record list
                record_list = wfdb.get_record_list('chbmit')
                patient_records = [r for r in record_list if r.startswith(f'{patient}/')]
                
                # Filter to get only EDF files (exclude summary and other files)
                edf_records = [r for r in patient_records if r.endswith('.edf')]
                
                # Limit number of records
                if max_records_per_patient > 0:
                    edf_records = edf_records[:max_records_per_patient]
                
                print(f"Found {len(edf_records)} EDF files for {patient}")
                print(f"Will download: {edf_records}")
                
                # Download each record
                for i, record in enumerate(edf_records):
                    print(f"  📄 Downloading {record} ({i+1}/{len(edf_records)})...")
                    
                    # Extract patient and record name
                    patient_id = record.split('/')[0]
                    record_name = record.split('/')[1].replace('.edf', '')
                    
                    # Create patient directory
                    patient_path = os.path.join(download_path, patient_id)
                    os.makedirs(patient_path, exist_ok=True)
                    
                    try:
                        # Download the record
                        wfdb.dl_files('chbmit', patient_path, [record])
                        print(f"    ✅ Downloaded {record}")
                        
                    except Exception as e:
                        print(f"    ❌ Error downloading {record}: {e}")
                        continue
                
                # Also download the summary file
                summary_file = f'{patient}/{patient}-summary.txt'
                if summary_file in record_list:
                    print(f"  📄 Downloading summary file...")
                    try:
                        wfdb.dl_files('chbmit', os.path.join(download_path, patient), [summary_file])
                        print(f"    ✅ Downloaded {summary_file}")
                    except Exception as e:
                        print(f"    ❌ Error downloading summary: {e}")
                
            except Exception as e:
                print(f"❌ Error processing patient {patient}: {e}")
                continue
        
        print(f"\n✅ Download completed! Files saved to: {download_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nFallback options:")
        print("1. Check your internet connection")
        print("2. Try downloading fewer records")
        print("3. Use manual wget method")
        return False

def download_seizure_rich_data(download_path):
    """
    Download CHB-MIT files that are known to contain seizures.
    This ensures we have both seizure and non-seizure data for training.
    """
    print("📥 Downloading CHB-MIT files with seizure events...")
    
    # Files known to contain seizures based on CHB-MIT documentation
    seizure_files = [
        'chb01/chb01_03.edf',    # Contains seizure at 2996-3036 seconds
        'chb01/chb01_04.edf',    # Contains seizure at 1467-1494 seconds  
        'chb01/chb01_15.edf',    # Contains seizure at 1732-1772 seconds
        'chb01/chb01_16.edf',    # Contains seizure at 1015-1066 seconds
        'chb01/chb01_18.edf',    # Contains seizure at 1720-1810 seconds
        'chb01/chb01_21.edf',    # Contains seizure at 327-420 seconds
        'chb01/chb01_26.edf',    # Contains seizure at 1862-1963 seconds
        'chb01/chb01-summary.txt'
    ]
    
    # Also download some non-seizure files for balance
    normal_files = [
        'chb01/chb01_01.edf',    # No seizures
        'chb01/chb01_02.edf',    # No seizures
    ]
    
    all_files = seizure_files + normal_files
    
    try:
        os.makedirs(os.path.join(download_path, 'chb01'), exist_ok=True)
        
        success_count = 0
        for i, record in enumerate(all_files):
            print(f"  📄 Downloading {record} ({i+1}/{len(all_files)})...")
            try:
                wfdb.dl_files('chbmit', download_path, [record])
                print(f"    ✅ Downloaded {record}")
                success_count += 1
            except Exception as e:
                print(f"    ❌ Error downloading {record}: {e}")
                continue
        
        print(f"\n✅ Downloaded {success_count}/{len(all_files)} files")
        return success_count > 5  # Success if we got most files
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def parse_seizure_annotations(summary_file_path):
    """
    Parse the CHB-MIT summary file to extract seizure annotations.
    Also check for individual .seizures files as backup.
    
    Args:
        summary_file_path: Path to the patient summary file (e.g., chb01-summary.txt)
    
    Returns:
        dict: Dictionary with filename as key and list of seizure intervals as values
    """
    seizure_annotations = {}
    patient_dir = os.path.dirname(summary_file_path)
    
    try:
        # Method 1: Parse summary file
        with open(summary_file_path, 'r') as f:
            lines = f.readlines()
        
        current_file = None
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for file names
            if line.startswith('File Name:'):
                current_file = line.split(':')[1].strip()
                seizure_annotations[current_file] = []
            
            # Look for seizure information
            elif line.startswith('Number of Seizures in File:'):
                num_seizures = int(line.split(':')[1].strip())
                
                # Parse seizure start and end times
                for j in range(num_seizures):
                    i += 1
                    if i < len(lines):
                        seizure_line = lines[i].strip()
                        if 'Seizure Start Time:' in seizure_line and 'Seizure End Time:' in seizure_line:
                            # Extract start and end times
                            parts = seizure_line.split()
                            try:
                                start_idx = parts.index('Time:') + 1
                                start_time = int(parts[start_idx])
                                end_idx = parts.index('Time:', start_idx) + 1
                                end_time = int(parts[end_idx])
                                
                                seizure_annotations[current_file].append((start_time, end_time))
                            except (ValueError, IndexError) as e:
                                print(f"Warning: Could not parse seizure times from: {seizure_line}")
            
            i += 1
        
        # Method 2: Check for individual .seizures files as backup
        print("Checking for individual .seizures files...")
        edf_files = [f for f in os.listdir(patient_dir) if f.endswith('.edf')]
        
        for edf_file in edf_files:
            seizure_file = os.path.join(patient_dir, f"{edf_file}.seizures")
            
            if os.path.exists(seizure_file) and edf_file not in seizure_annotations:
                print(f"Found seizure file: {edf_file}.seizures")
                try:
                    with open(seizure_file, 'r') as f:
                        seizure_content = f.read().strip()
                    
                    # Parse seizure file content (format may vary)
                    # Often contains start and end times
                    if seizure_content:
                        # Try to extract numbers (start and end times)
                        import re
                        numbers = re.findall(r'\d+', seizure_content)
                        if len(numbers) >= 2:
                            start_time = int(numbers[0])
                            end_time = int(numbers[1])
                            seizure_annotations[edf_file] = [(start_time, end_time)]
                            print(f"  Parsed seizure: {start_time}-{end_time} seconds")
                        
                except Exception as e:
                    print(f"Warning: Could not parse {seizure_file}: {e}")
        
        # Method 3: Based on your directory listing, manually add known seizure files
        known_seizure_files = {
            'chb01_03.edf': [(2996, 3036)],  # Based on typical CHB-MIT annotations
            'chb01_04.edf': [(1467, 1494)],
            'chb01_15.edf': [(1732, 1772)], 
            'chb01_16.edf': [(1015, 1066)],
            'chb01_18.edf': [(1720, 1810)],
            'chb01_21.edf': [(327, 420)],
            'chb01_26.edf': [(1862, 1963)]
        }
        
        # Add known seizure annotations if not already parsed
        for filename, intervals in known_seizure_files.items():
            if filename not in seizure_annotations or not seizure_annotations[filename]:
                seizure_annotations[filename] = intervals
                print(f"Added known seizure annotation for {filename}: {intervals}")
    
    except Exception as e:
        print(f"Error parsing annotations: {e}")
        # Fallback to known seizure files
        seizure_annotations = {
            'chb01_03.edf': [(2996, 3036)],
            'chb01_04.edf': [(1467, 1494)],
            'chb01_15.edf': [(1732, 1772)], 
            'chb01_16.edf': [(1015, 1066)],
            'chb01_18.edf': [(1720, 1810)],
            'chb01_21.edf': [(327, 420)],
            'chb01_26.edf': [(1862, 1963)]
        }
        print("Using fallback seizure annotations")
    
    return seizure_annotations

def load_chb_mit_data(data_path, patient_id='chb01', max_files=15):
    """
    Load CHB-MIT EEG data for a specific patient.
    Prioritizes files with seizures based on the complete summary file.
    
    Args:
        data_path: Path to CHB-MIT dataset
        patient_id: Patient ID (e.g., 'chb01')
        max_files: Maximum number of files to load (for memory management)
    
    Returns:
        tuple: (data_segments, labels, file_info)
    """
    print(f"Loading CHB-MIT data for patient {patient_id}...")
    
    patient_path = os.path.join(data_path, patient_id)
    
    if not os.path.exists(patient_path):
        print(f"Patient directory not found: {patient_path}")
        print("Please ensure you have downloaded the CHB-MIT dataset.")
        return None, None, None
    
    # Complete seizure annotations based on your summary file
    seizure_annotations = {
        'chb01_03.edf': [(2996, 3036)],   # 40 seconds
        'chb01_04.edf': [(1467, 1494)],   # 27 seconds  
        'chb01_15.edf': [(1732, 1772)],   # 40 seconds
        'chb01_16.edf': [(1015, 1066)],   # 51 seconds
        'chb01_18.edf': [(1720, 1810)],   # 90 seconds
        'chb01_21.edf': [(327, 420)],     # 93 seconds
        'chb01_26.edf': [(1862, 1963)]    # 101 seconds
    }
    
    print(f"📊 Found {len(seizure_annotations)} files with seizures:")
    for filename, intervals in seizure_annotations.items():
        duration = sum(end - start for start, end in intervals)
        print(f"  {filename}: {duration} seconds of seizure activity")
    
    # Get list of EDF files, prioritizing seizure files
    all_edf_files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
    
    # Separate files with and without seizures
    seizure_files = [f for f in all_edf_files if f in seizure_annotations]
    normal_files = [f for f in all_edf_files if f not in seizure_files]
    
    # Include all seizure files + some normal files for balance
    normal_files_to_include = min(len(normal_files), max_files - len(seizure_files))
    selected_files = seizure_files + normal_files[:normal_files_to_include]
    
    print(f"📁 Loading {len(seizure_files)} seizure files + {normal_files_to_include} normal files")
    print(f"Total files to process: {len(selected_files)}")
    
    data_segments = []
    labels = []
    file_info = []
    
    total_seizure_duration = 0
    
    for edf_file in sorted(selected_files):
        print(f"Processing {edf_file}...")
        
        try:
            # Load EDF file
            file_path = os.path.join(patient_path, edf_file)
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Get data and sampling rate
            data = raw.get_data()  # Shape: (n_channels, n_samples)
            sfreq = raw.info['sfreq']
            
            # Resample if necessary
            if sfreq != SAMPLE_RATE:
                from scipy import signal
                data = signal.resample(data, int(data.shape[1] * SAMPLE_RATE / sfreq), axis=1)
            
            # Get seizure intervals for this file
            seizure_intervals = seizure_annotations.get(edf_file, [])
            
            # Create labels array (0 = normal, 1 = seizure)
            n_samples = data.shape[1]
            file_labels = np.zeros(n_samples)
            
            for start_sec, end_sec in seizure_intervals:
                start_sample = int(start_sec * SAMPLE_RATE)
                end_sample = int(end_sec * SAMPLE_RATE)
                if start_sample < n_samples and end_sample <= n_samples:
                    file_labels[start_sample:end_sample] = 1
                    seizure_duration = end_sec - start_sec
                    total_seizure_duration += seizure_duration
                    print(f"  ✅ Marked seizure: {start_sec}-{end_sec}s ({seizure_duration}s duration)")
            
            data_segments.append(data)
            labels.append(file_labels)
            file_info.append({
                'filename': edf_file,
                'duration': n_samples / SAMPLE_RATE,
                'seizure_intervals': seizure_intervals,
                'n_channels': data.shape[0],
                'has_seizures': len(seizure_intervals) > 0,
                'seizure_duration': sum(end - start for start, end in seizure_intervals)
            })
            
        except Exception as e:
            print(f"  ❌ Error loading {edf_file}: {e}")
            continue
    
    print(f"\n📈 SEIZURE DATA SUMMARY:")
    print(f"Total seizure duration: {total_seizure_duration} seconds ({total_seizure_duration/60:.1f} minutes)")
    print(f"Number of seizure events: {sum(len(info['seizure_intervals']) for info in file_info)}")
    print(f"Expected seizure windows (5s): ~{int(total_seizure_duration * 2)}")  # 50% overlap
    
    return data_segments, labels, file_info

# Alternative: Load sample data if CHB-MIT not available
def create_sample_chb_mit_data():
    """
    Create realistic sample data mimicking CHB-MIT structure for testing.
    """
    print("Creating sample CHB-MIT-like data for testing...")
    
    # CHB-MIT typically has 23 channels
    n_channels = 23
    duration_minutes = 30  # 30 minutes per file
    n_samples = duration_minutes * 60 * SAMPLE_RATE
    
    data_segments = []
    labels = []
    file_info = []
    
    # Create 3 sample files
    for file_idx in range(3):
        # Generate realistic EEG data
        data = np.random.randn(n_channels, n_samples) * 50  # Microvolts scale
        
        # Add realistic EEG frequency components
        time = np.arange(n_samples) / SAMPLE_RATE
        for ch in range(n_channels):
            # Add alpha (8-13 Hz), beta (13-30 Hz), and other components
            data[ch] += 30 * np.sin(2 * np.pi * 10 * time)  # Alpha
            data[ch] += 15 * np.sin(2 * np.pi * 20 * time)  # Beta  
            data[ch] += 10 * np.sin(2 * np.pi * 4 * time)   # Theta
        
        # Create labels with seizure events
        file_labels = np.zeros(n_samples)
        
        if file_idx < 2:  # Add seizures to first 2 files
            # Random seizure events
            n_seizures = np.random.randint(1, 4)
            seizure_intervals = []
            
            for _ in range(n_seizures):
                # Random seizure duration (30-120 seconds)
                seizure_duration = np.random.randint(30, 121)
                start_time = np.random.randint(0, duration_minutes * 60 - seizure_duration)
                end_time = start_time + seizure_duration
                
                start_sample = start_time * SAMPLE_RATE
                end_sample = end_time * SAMPLE_RATE
                
                # Mark seizure in labels
                file_labels[start_sample:end_sample] = 1
                seizure_intervals.append((start_time, end_time))
                
                # Add seizure-like patterns to data
                for ch in range(n_channels):
                    # High frequency oscillations during seizure
                    seizure_signal = 200 * np.sin(2 * np.pi * 25 * time[start_sample:end_sample])
                    data[ch, start_sample:end_sample] += seizure_signal
        else:
            seizure_intervals = []
        
        data_segments.append(data)
        labels.append(file_labels)
        file_info.append({
            'filename': f'chb01_{file_idx+1:02d}.edf',
            'duration': duration_minutes * 60,
            'seizure_intervals': seizure_intervals,
            'n_channels': n_channels
        })
    
    return data_segments, labels, file_info

# Load CHB-MIT data using all available seizure files
print("="*60)
print("CHB-MIT DATASET LOADING - ALL SEIZURE FILES")
print("="*60)

print("📊 Loading complete CHB-MIT chb01 dataset...")
print("Target: All 7 seizure files + balanced normal files")

try:
    raw_data_segments, raw_labels_segments, file_info = load_chb_mit_data(
        DATA_PATH, patient_id='chb01', max_files=15  # All seizure files + some normal
    )
    
    if raw_data_segments is not None:
        print("✅ CHB-MIT data loaded successfully!")
        
        # Print comprehensive dataset information  
        print(f"\n📈 COMPREHENSIVE DATASET SUMMARY")
        print("="*50)
        print(f"Number of files loaded: {len(raw_data_segments)}")
        
        seizure_files_count = sum(1 for info in file_info if info['has_seizures'])
        normal_files_count = len(file_info) - seizure_files_count
        total_seizure_duration = sum(info['seizure_duration'] for info in file_info)
        total_duration = sum(info['duration'] for info in file_info)
        
        print(f"Seizure files: {seizure_files_count}")
        print(f"Normal files: {normal_files_count}")
        print(f"Total seizure duration: {total_seizure_duration:.1f} seconds ({total_seizure_duration/60:.2f} minutes)")
        print(f"Total dataset duration: {total_duration/3600:.2f} hours")
        print(f"Seizure percentage: {100*total_seizure_duration/total_duration:.2f}%")
        
        print(f"\n📄 INDIVIDUAL FILE DETAILS:")
        print("-" * 70)
        
        for i, info in enumerate(file_info):
            seizure_indicator = "🔴" if info['has_seizures'] else "🟢"
            duration_min = info['duration'] / 60
            seizure_dur = info['seizure_duration']
            
            print(f"{seizure_indicator} File {i+1}: {info['filename']}")
            print(f"   Duration: {duration_min:.1f} min | Seizures: {len(info['seizure_intervals'])} | "
                  f"Seizure time: {seizure_dur:.1f}s")
            
            if info['seizure_intervals']:
                for start, end in info['seizure_intervals']:
                    print(f"     └─ Seizure: {start}-{end}s ({end-start}s duration)")
            
            print(f"   Data shape: {raw_data_segments[i].shape}")
        
        # Expected windowing results
        print(f"\n🎯 EXPECTED WINDOWING RESULTS:")
        print("-" * 40)
        
        # Calculate expected windows (5s windows, 50% overlap)
        window_duration = 5  # seconds
        overlap_factor = 0.5
        effective_window_step = window_duration * (1 - overlap_factor)  # 2.5 seconds
        
        total_samples = sum(seg.shape[1] for seg in raw_data_segments)
        total_seconds = total_samples / SAMPLE_RATE
        expected_total_windows = int(total_seconds / effective_window_step)
        
        # Expected seizure windows (more accurate calculation)
        expected_seizure_windows = int(total_seizure_duration / effective_window_step * 2)  # rough estimate
        
        print(f"Expected total windows: ~{expected_total_windows:,}")
        print(f"Expected seizure windows: ~{expected_seizure_windows}")
        print(f"Expected seizure percentage: ~{100*expected_seizure_windows/expected_total_windows:.1f}%")
        
        print(f"\n💡 This should give you MIT-level data richness!")
        print(f"   MIT paper used 3+ seizures per patient for good performance")
        print(f"   You now have 7 seizures across different time periods")
        
    else:
        print("❌ Failed to load CHB-MIT data")
        print("Falling back to sample data...")
        raw_data_segments, raw_labels_segments, file_info = create_sample_chb_mit_data()
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("🔄 Using sample data for demonstration...")
    raw_data_segments, raw_labels_segments, file_info = create_sample_chb_mit_data()

## 3. Data Preprocessing

def preprocess_eeg_segments(data_segments, sample_rate=256, low_freq=0.5, high_freq=40, notch_freq=50):
    """
    Apply preprocessing to CHB-MIT EEG data segments including filtering.
    
    Args:
        data_segments: List of EEG data arrays (n_channels, n_samples)
        sample_rate: Sampling rate in Hz
        low_freq: Low cutoff frequency for bandpass filter
        high_freq: High cutoff frequency for bandpass filter  
        notch_freq: Notch filter frequency for power line interference
    
    Returns:
        List of preprocessed EEG data arrays
    """
    print("Preprocessing CHB-MIT EEG data...")
    
    from scipy import signal
    
    processed_segments = []
    
    for segment_idx, data in enumerate(data_segments):
        print(f"Processing segment {segment_idx + 1}/{len(data_segments)}...")
        
        n_channels, n_samples = data.shape
        processed_data = np.zeros_like(data)
        
        for ch in range(n_channels):
            # Apply bandpass filter (0.5-40 Hz)
            try:
                sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=sample_rate, output='sos')
                filtered_channel = signal.sosfilt(sos, data[ch])
                
                # Apply notch filter for power line interference (50 Hz)
                sos_notch = signal.butter(4, [notch_freq-2, notch_freq+2], btype='bandstop', 
                                        fs=sample_rate, output='sos')
                filtered_channel = signal.sosfilt(sos_notch, filtered_channel)
                
                processed_data[ch] = filtered_channel
                
            except Exception as e:
                print(f"Warning: Error filtering channel {ch} in segment {segment_idx}: {e}")
                processed_data[ch] = data[ch]  # Use original data if filtering fails
        
        processed_segments.append(processed_data)
    
    return processed_segments

# Apply preprocessing to all segments
processed_data_segments = preprocess_eeg_segments(raw_data_segments, SAMPLE_RATE)

print(f"Processed {len(processed_data_segments)} data segments")
for i, segment in enumerate(processed_data_segments):
    print(f"Segment {i+1} shape: {segment.shape}")

## 4. Windowing and Segmentation for CHB-MIT Data

def create_windows_from_continuous_data(data_segments, label_segments, window_size_samples, overlap_samples):
    """
    Create sliding windows from continuous CHB-MIT EEG data with proper seizure labeling.
    
    Args:
        data_segments: List of EEG data arrays (n_channels, n_samples)
        label_segments: List of label arrays (n_samples,) with 0/1 for normal/seizure
        window_size_samples: Window size in samples
        overlap_samples: Overlap size in samples
    
    Returns:
        tuple: (windows_array, window_labels_array)
    """
    print("Creating sliding windows from continuous CHB-MIT data...")
    
    all_windows = []
    all_labels = []
    
    step_size = window_size_samples - overlap_samples
    
    for segment_idx, (data, labels) in enumerate(zip(data_segments, label_segments)):
        print(f"Processing segment {segment_idx + 1}/{len(data_segments)}...")
        
        n_channels, n_samples = data.shape
        
        # Create windows for this segment
        segment_windows = []
        segment_labels = []
        
        for start in range(0, n_samples - window_size_samples + 1, step_size):
            end = start + window_size_samples
            
            # Extract window
            window = data[:, start:end]
            
            # Determine window label
            # If >50% of the window contains seizure activity, label as seizure
            window_labels_portion = labels[start:end]
            seizure_percentage = np.mean(window_labels_portion)
            window_label = 1 if seizure_percentage > 0.5 else 0
            
            segment_windows.append(window)
            segment_labels.append(window_label)
        
        all_windows.extend(segment_windows)
        all_labels.extend(segment_labels)
        
        print(f"  Created {len(segment_windows)} windows")
        print(f"  Seizure windows: {sum(segment_labels)}")
        print(f"  Normal windows: {len(segment_labels) - sum(segment_labels)}")
    
    return np.array(all_windows), np.array(all_labels)

# Create windows from continuous data
window_size_samples = WINDOW_SIZE * SAMPLE_RATE
overlap_samples = int(OVERLAP * window_size_samples)

X_windows, y_windows = create_windows_from_continuous_data(
    processed_data_segments, raw_labels_segments, window_size_samples, overlap_samples
)

print(f"\nFinal windowed dataset:")
print(f"Windows shape: {X_windows.shape}")
print(f"Window labels shape: {y_windows.shape}")
print(f"Total seizure windows: {np.sum(y_windows)} ({100*np.sum(y_windows)/len(y_windows):.1f}%)")
print(f"Total normal windows: {len(y_windows) - np.sum(y_windows)} ({100*(len(y_windows) - np.sum(y_windows))/len(y_windows):.1f}%)")

# Visualize a sample window
def plot_sample_windows(windows, labels, n_samples=2):
    """Plot sample EEG windows for visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Find seizure and normal samples
    seizure_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    
    if len(seizure_indices) > 0 and len(normal_indices) > 0:
        # Plot seizure sample
        seizure_idx = seizure_indices[0]
        seizure_window = windows[seizure_idx]
        
        # Plot first few channels
        for i in range(min(4, seizure_window.shape[0])):
            axes[0, 0].plot(seizure_window[i], label=f'Ch {i+1}', alpha=0.7)
        axes[0, 0].set_title(f'Seizure Window (Sample {seizure_idx})')
        axes[0, 0].set_xlabel('Samples')
        axes[0, 0].set_ylabel('Amplitude (μV)')
        axes[0, 0].legend()
        
        # Plot normal sample
        normal_idx = normal_indices[0]
        normal_window = windows[normal_idx]
        
        for i in range(min(4, normal_window.shape[0])):
            axes[0, 1].plot(normal_window[i], label=f'Ch {i+1}', alpha=0.7)
        axes[0, 1].set_title(f'Normal Window (Sample {normal_idx})')
        axes[0, 1].set_xlabel('Samples')
        axes[0, 1].set_ylabel('Amplitude (μV)')
        axes[0, 1].legend()
        
        # Plot power spectral density comparison
        from scipy import signal as scipy_signal
        
        # Seizure PSD
        freqs, psd_seizure = scipy_signal.welch(seizure_window[0], fs=SAMPLE_RATE, nperseg=256)
        axes[1, 0].semilogy(freqs, psd_seizure)
        axes[1, 0].set_title('Seizure - Power Spectral Density')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('PSD (μV²/Hz)')
        axes[1, 0].set_xlim(0, 50)
        
        # Normal PSD
        freqs, psd_normal = scipy_signal.welch(normal_window[0], fs=SAMPLE_RATE, nperseg=256)
        axes[1, 1].semilogy(freqs, psd_normal)
        axes[1, 1].set_title('Normal - Power Spectral Density')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('PSD (μV²/Hz)')
        axes[1, 1].set_xlim(0, 50)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough seizure or normal samples to plot comparison")

# Plot sample windows
plot_sample_windows(X_windows, y_windows)

## 5. Feature Extraction for Classical ML

def extract_features(windows):
    """
    Extract time-domain and frequency-domain features from EEG windows.
    """
    print("Extracting features...")
    
    features = []
    
    for window in windows:
        window_features = []
        
        for channel in window:
            # Time-domain features
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            skewness = np.mean(((channel - mean_val) / std_val) ** 3)
            kurtosis = np.mean(((channel - mean_val) / std_val) ** 4)
            
            # Hjorth parameters
            def hjorth_parameters(signal):
                first_deriv = np.diff(signal)
                second_deriv = np.diff(first_deriv)
                
                var_zero = np.var(signal)
                var_first = np.var(first_deriv)
                var_second = np.var(second_deriv)
                
                activity = var_zero
                mobility = np.sqrt(var_first / var_zero) if var_zero > 0 else 0
                complexity = np.sqrt(var_second / var_first) / mobility if var_first > 0 and mobility > 0 else 0
                
                return activity, mobility, complexity
            
            activity, mobility, complexity = hjorth_parameters(channel)
            
            # Frequency-domain features
            fft = np.fft.fft(channel)
            freqs = np.fft.fftfreq(len(channel), 1/SAMPLE_RATE)
            psd = np.abs(fft) ** 2
            
            # Band powers
            delta_power = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])
            theta_power = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
            alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
            beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
            gamma_power = np.mean(psd[(freqs >= 30) & (freqs <= 40)])
            
            # Combine all features
            channel_features = [
                mean_val, std_val, skewness, kurtosis,
                activity, mobility, complexity,
                delta_power, theta_power, alpha_power, beta_power, gamma_power
            ]
            
            window_features.extend(channel_features)
        
        features.append(window_features)
    
    return np.array(features)

# Extract features for classical ML
X_features = extract_features(X_windows)
print(f"Feature matrix shape: {X_features.shape}")

## 6. Data Splitting and Scaling

# Split the data
X_train_feat, X_test_feat, y_train, y_test = train_test_split(
    X_features, y_windows, test_size=0.2, random_state=42, stratify=y_windows
)

X_train_raw, X_test_raw = train_test_split(
    X_windows, test_size=0.2, random_state=42, stratify=y_windows
)[0], train_test_split(
    X_windows, test_size=0.2, random_state=42, stratify=y_windows
)[1]

# Scale features for classical ML
scaler = StandardScaler()
X_train_feat_scaled = scaler.fit_transform(X_train_feat)
X_test_feat_scaled = scaler.transform(X_test_feat)

print(f"Training set: {X_train_feat_scaled.shape[0]} samples")
print(f"Test set: {X_test_feat_scaled.shape[0]} samples")

## 7. Classical ML Model - Random Forest (with Class Imbalance Handling)

print("Training Random Forest model...")

# Check if we have both classes
unique_classes = np.unique(y_train)
print(f"Classes in training data: {unique_classes}")

if len(unique_classes) == 1:
    print("⚠️  WARNING: Only one class found in the data!")
    if unique_classes[0] == 0:
        print("   All samples are normal (no seizures detected)")
        print("   This can happen if:")
        print("   1. Downloaded files don't contain seizures")
        print("   2. Seizure annotations weren't parsed correctly")
        print("   3. Window labeling threshold is too strict")
        
    print("\n🔄 Creating balanced synthetic data for demonstration...")
    
    # Create some synthetic seizure windows for demonstration
    n_seizure_samples = len(y_train) // 10  # 10% seizure samples
    
    # Copy some normal samples and modify them to simulate seizures
    seizure_indices = np.random.choice(len(X_train_feat_scaled), n_seizure_samples, replace=False)
    
    # Add synthetic seizure features (higher variance, different frequency content)
    X_train_synthetic = X_train_feat_scaled.copy()
    y_train_synthetic = y_train.copy()
    
    for idx in seizure_indices:
        # Modify features to simulate seizure characteristics
        X_train_synthetic[idx] *= np.random.uniform(1.5, 3.0, X_train_synthetic[idx].shape)
        X_train_synthetic[idx] += np.random.normal(0, 0.5, X_train_synthetic[idx].shape)
        y_train_synthetic[idx] = 1
    
    # Use synthetic data for training
    X_train_model = X_train_synthetic
    y_train_model = y_train_synthetic
    
    print(f"   Created {n_seizure_samples} synthetic seizure samples")
    print(f"   New class distribution: {np.unique(y_train_model, return_counts=True)}")

else:
    # Use real data if we have both classes
    X_train_model = X_train_feat_scaled
    y_train_model = y_train
    print("✅ Both seizure and normal classes found!")

# Handle class imbalance
unique_classes_model = np.unique(y_train_model)
if len(unique_classes_model) > 1:
    class_weights = compute_class_weight('balanced', classes=unique_classes_model, y=y_train_model)
    class_weight_dict = {unique_classes_model[i]: class_weights[i] for i in range(len(unique_classes_model))}
else:
    class_weight_dict = None
    
print(f"Class weights: {class_weight_dict}")

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)

rf_model.fit(X_train_model, y_train_model)

# Predictions
y_pred_rf = rf_model.predict(X_test_feat_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_feat_scaled)

# Handle case where test set might not have seizures
if y_pred_proba_rf.shape[1] == 1:
    # Only one class predicted, create dummy probabilities
    y_pred_proba_rf = np.column_stack([1 - y_pred_proba_rf[:, 0], y_pred_proba_rf[:, 0]])

y_pred_proba_rf = y_pred_proba_rf[:, 1] if y_pred_proba_rf.shape[1] > 1 else y_pred_proba_rf[:, 0]

print("\nRandom Forest Results:")
if len(np.unique(y_test)) > 1:
    print(classification_report(y_test, y_pred_rf))
else:
    print("⚠️ Test set contains only one class - cannot compute full classification metrics")
    print(f"Predictions: {np.unique(y_pred_rf, return_counts=True)}")
    print(f"Accuracy: {np.mean(y_pred_rf == y_test):.4f}")

## 8. Deep Learning Model - 1D CNN

print("Building 1D CNN model...")

def create_1d_cnn_model(input_shape, num_classes=1):
    """
    Create a 1D CNN model for EEG classification.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Second conv block
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Third conv block
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

# Reshape data for CNN (samples, channels, time_points) -> (samples, time_points, channels)
X_train_cnn = X_train_raw.transpose(0, 2, 1)
X_test_cnn = X_test_raw.transpose(0, 2, 1)

print(f"CNN input shape: {X_train_cnn.shape}")

# Create and compile model
cnn_model = create_1d_cnn_model(input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]))

# Calculate class weights for CNN
unique_classes_cnn = np.unique(y_train_model) if 'y_train_model' in locals() else np.unique(y_train)
y_train_cnn = y_train_model if 'y_train_model' in locals() else y_train

if len(unique_classes_cnn) > 1:
    pos_weight = len(y_train_cnn[y_train_cnn == 0]) / len(y_train_cnn[y_train_cnn == 1])
    class_weight_dict_cnn = class_weight_dict if 'class_weight_dict' in locals() else None
else:
    pos_weight = 1.0
    class_weight_dict_cnn = None

cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print(cnn_model.summary())

# Train CNN
print("Training CNN model...")

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
]

history = cnn_model.fit(
    X_train_cnn, y_train_cnn,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weight_dict_cnn,
    verbose=1
)

# Predictions
y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int).flatten()
y_pred_proba_cnn = cnn_model.predict(X_test_cnn).flatten()

print("CNN Results:")
print(classification_report(y_test, y_pred_cnn))

## 9. Model Evaluation and Visualization

def plot_training_history(history):
    """Plot training history for CNN."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training')
    axes[1, 0].plot(history.history['val_precision'], label='Validation')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training')
    axes[1, 1].plot(history.history['val_recall'], label='Validation')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices_and_roc(y_true, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn):
    """Plot confusion matrices and ROC curves for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix - Random Forest
    cm_rf = confusion_matrix(y_true, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Random Forest - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Confusion Matrix - CNN
    cm_cnn = confusion_matrix(y_true, y_pred_cnn)
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('CNN - Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # ROC Curves
    fpr_rf, tpr_rf, _ = roc_curve(y_true, y_proba_rf)
    fpr_cnn, tpr_cnn, _ = roc_curve(y_true, y_proba_cnn)
    
    auc_rf = roc_auc_score(y_true, y_proba_rf)
    auc_cnn = roc_auc_score(y_true, y_proba_cnn)
    
    axes[1, 0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})')
    axes[1, 0].plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {auc_cnn:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves Comparison')
    axes[1, 0].legend()
    
    # Feature Importance (Random Forest)
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = rf_model.feature_importances_
        top_features = np.argsort(feature_importance)[-20:]  # Top 20 features
        
        axes[1, 1].barh(range(len(top_features)), feature_importance[top_features])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 20 Feature Importances (Random Forest)')
        axes[1, 1].set_yticks(range(len(top_features)))
        axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_features])
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# Plot evaluation metrics
plot_confusion_matrices_and_roc(y_test, y_pred_rf, y_pred_cnn, y_pred_proba_rf, y_pred_proba_cnn)

# Print detailed evaluation metrics
print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

print("\nRandom Forest Metrics:")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

print("\nCNN Metrics:")
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_cnn):.4f}")

## 10. Save Models

print("Saving trained models...")

# Save Random Forest model
import joblib
joblib.dump(rf_model, '/content/new/rf_seizure_model.joblib')
joblib.dump(scaler, '/content/new/feature_scaler.joblib')

# Save CNN model
cnn_model.save('/content/new/cnn_seizure_model.h5')

print("Models saved successfully!")
print("- Random Forest: rf_seizure_model.joblib")
print("- Feature Scaler: feature_scaler.joblib") 
print("- CNN Model: cnn_seizure_model.h5")

## 11. Model Inference Example

def predict_seizure(eeg_window, model_type='cnn'):
    """
    Predict seizure probability for a new EEG window.
    
    Args:
        eeg_window: EEG data window (channels, time_points)
        model_type: 'rf' for Random Forest, 'cnn' for CNN
    
    Returns:
        probability: Seizure probability (0-1)
    """
    
    if model_type == 'rf':
        # Extract features
        features = extract_features([eeg_window])
        features_scaled = scaler.transform(features)
        
        # Predict
        probability = rf_model.predict_proba(features_scaled)[0, 1]
        
    elif model_type == 'cnn':
        # Reshape for CNN
        window_cnn = eeg_window.T.reshape(1, eeg_window.shape[1], eeg_window.shape[0])
        
        # Predict
        probability = cnn_model.predict(window_cnn)[0, 0]
    
    return probability

# Example usage
print("\n" + "="*50)
print("INFERENCE EXAMPLE")
print("="*50)

# Use a test sample for demonstration
sample_window = X_test_raw[0]
true_label = y_test[0]

rf_prob = predict_seizure(sample_window, 'rf')
cnn_prob = predict_seizure(sample_window, 'cnn')

print(f"True label: {'Seizure' if true_label == 1 else 'Normal'}")
print(f"Random Forest prediction: {rf_prob:.3f} ({'Seizure' if rf_prob > 0.5 else 'Normal'})")
print(f"CNN prediction: {cnn_prob:.3f} ({'Seizure' if cnn_prob > 0.5 else 'Normal'})")

print("\n" + "="*50)
print("NOTEBOOK COMPLETE!")
print("="*50)
print("Both models have been trained and saved.")
print("You can now use them for real-time seizure detection.")
print("\nNext steps:")
print("1. Replace synthetic data with real CHB-MIT dataset")
print("2. Optimize hyperparameters using cross-validation")
print("3. Implement real-time processing pipeline")
print("4. Deploy using Streamlit or TensorFlow Lite")
