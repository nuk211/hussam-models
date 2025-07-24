# Comprehensive Model Testing Suite for CHB-MIT Seizure Classification
# Test current models thoroughly before scaling up

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, f1_score)
import joblib
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

## 1. Load Your Trained Models

def load_trained_models():
    """
    Load the models you just trained.
    """
    print("📁 Loading your trained models...")
    
    try:
        # Load Random Forest model
        rf_model = joblib.load('/content/new/rf_seizure_model.joblib')
        scaler = joblib.load('/content/new/feature_scaler.joblib')
        print("✅ Random Forest model loaded")
        
        # Load CNN model
        cnn_model = keras.models.load_model('/content/new/cnn_seizure_model.h5')
        print("✅ CNN model loaded")
        
        return rf_model, scaler, cnn_model
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please ensure models are saved correctly")
        return None, None, None

## 2. Comprehensive Performance Evaluation

def comprehensive_evaluation(y_true, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn):
    """
    Comprehensive evaluation of both models with multiple metrics.
    """
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    results = {}
    
    # Basic metrics for both models
    models = {
        'Random Forest': (y_pred_rf, y_proba_rf),
        'CNN': (y_pred_cnn, y_proba_cnn)
    }
    
    for model_name, (y_pred, y_proba) in models.items():
        print(f"\n📊 {model_name} Results:")
        print("-" * 40)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))
        
        # Additional metrics
        auc_score = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
        avg_precision = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0
        
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        
        # Store results
        results[model_name] = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1_score': report['1']['f1-score'] if '1' in report else 0,
            'auc_roc': auc_score,
            'avg_precision': avg_precision
        }
    
    return results

## 3. Visualization Suite

def create_comprehensive_visualizations(y_true, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn):
    """
    Create comprehensive visualizations for model evaluation.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Confusion Matrices
    cm_rf = confusion_matrix(y_true, y_pred_rf)
    cm_cnn = confusion_matrix(y_true, y_pred_cnn)
    
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Random Forest - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('CNN - Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 2. ROC Curves
    if len(np.unique(y_true)) > 1:
        fpr_rf, tpr_rf, _ = roc_curve(y_true, y_proba_rf)
        fpr_cnn, tpr_cnn, _ = roc_curve(y_true, y_proba_cnn)
        
        auc_rf = roc_auc_score(y_true, y_proba_rf)
        auc_cnn = roc_auc_score(y_true, y_proba_cnn)
        
        axes[1, 0].plot(fpr_rf, tpr_rf, label=f'RF (AUC = {auc_rf:.3f})', linewidth=2)
        axes[1, 0].plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {auc_cnn:.3f})', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curves
    if len(np.unique(y_true)) > 1:
        precision_rf, recall_rf, _ = precision_recall_curve(y_true, y_proba_rf)
        precision_cnn, recall_cnn, _ = precision_recall_curve(y_true, y_proba_cnn)
        
        ap_rf = average_precision_score(y_true, y_proba_rf)
        ap_cnn = average_precision_score(y_true, y_proba_cnn)
        
        axes[1, 1].plot(recall_rf, precision_rf, label=f'RF (AP = {ap_rf:.3f})', linewidth=2)
        axes[1, 1].plot(recall_cnn, precision_cnn, label=f'CNN (AP = {ap_cnn:.3f})', linewidth=2)
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 4. Prediction Distribution
    axes[2, 0].hist(y_proba_rf[y_true == 0], bins=30, alpha=0.7, label='Normal', color='blue')
    axes[2, 0].hist(y_proba_rf[y_true == 1], bins=30, alpha=0.7, label='Seizure', color='red')
    axes[2, 0].set_xlabel('Predicted Probability')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Random Forest - Probability Distribution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].hist(y_proba_cnn[y_true == 0], bins=30, alpha=0.7, label='Normal', color='blue')
    axes[2, 1].hist(y_proba_cnn[y_true == 1], bins=30, alpha=0.7, label='Seizure', color='red')
    axes[2, 1].set_xlabel('Predicted Probability')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('CNN - Probability Distribution')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

## 4. Real-World Performance Simulation

def simulate_real_world_performance(rf_model, scaler, cnn_model, test_data, test_labels):
    """
    Simulate real-world deployment scenarios.
    """
    print("\n" + "="*60)
    print("🌍 REAL-WORLD PERFORMANCE SIMULATION")
    print("="*60)
    
    # Scenario 1: Different probability thresholds
    print("\n📊 Testing Different Decision Thresholds:")
    print("-" * 50)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Get probabilities for test data
    # Extract features for RF
    test_features = extract_features(test_data)
    test_features_scaled = scaler.transform(test_features)
    rf_probabilities = rf_model.predict_proba(test_features_scaled)[:, 1]
    
    # CNN probabilities
    test_data_cnn = test_data.transpose(0, 2, 1)
    cnn_probabilities = cnn_model.predict(test_data_cnn).flatten()
    
    threshold_results = []
    
    for threshold in thresholds:
        # RF predictions with threshold
        rf_pred_thresh = (rf_probabilities >= threshold).astype(int)
        rf_f1 = f1_score(test_labels, rf_pred_thresh) if len(np.unique(test_labels)) > 1 else 0
        
        # CNN predictions with threshold
        cnn_pred_thresh = (cnn_probabilities >= threshold).astype(int)
        cnn_f1 = f1_score(test_labels, cnn_pred_thresh) if len(np.unique(test_labels)) > 1 else 0
        
        threshold_results.append({
            'threshold': threshold,
            'rf_f1': rf_f1,
            'cnn_f1': cnn_f1
        })
        
        print(f"Threshold {threshold:.1f}: RF F1={rf_f1:.3f}, CNN F1={cnn_f1:.3f}")
    
    # Find optimal thresholds
    df_thresh = pd.DataFrame(threshold_results)
    optimal_rf_thresh = df_thresh.loc[df_thresh['rf_f1'].idxmax(), 'threshold']
    optimal_cnn_thresh = df_thresh.loc[df_thresh['cnn_f1'].idxmax(), 'threshold']
    
    print(f"\n🎯 Optimal Thresholds:")
    print(f"Random Forest: {optimal_rf_thresh:.1f}")
    print(f"CNN: {optimal_cnn_thresh:.1f}")
    
    return threshold_results, optimal_rf_thresh, optimal_cnn_thresh

## 5. Model Sufficiency Assessment

def assess_model_sufficiency(results, threshold_results):
    """
    Determine if current models are sufficient for deployment.
    """
    print("\n" + "="*60)
    print("✅ MODEL SUFFICIENCY ASSESSMENT")
    print("="*60)
    
    # Define minimum requirements for seizure detection
    min_requirements = {
        'sensitivity_recall': 0.80,  # Must catch 80% of seizures
        'precision': 0.70,           # 70% of seizure predictions should be correct
        'auc_roc': 0.85,            # Good discrimination ability
        'false_positive_rate': 0.10  # Max 10% false positive rate
    }
    
    print("📋 Minimum Requirements for Clinical Use:")
    for req, value in min_requirements.items():
        print(f"  {req.replace('_', ' ').title()}: {value}")
    
    print("\n📊 Current Model Performance:")
    
    assessment = {}
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        
        meets_requirements = True
        
        # Check sensitivity (recall)
        sensitivity = model_results['recall']
        meets_sens = sensitivity >= min_requirements['sensitivity_recall']
        print(f"  Sensitivity: {sensitivity:.3f} {'✅' if meets_sens else '❌'} (≥{min_requirements['sensitivity_recall']})")
        
        # Check precision
        precision = model_results['precision']
        meets_prec = precision >= min_requirements['precision']
        print(f"  Precision: {precision:.3f} {'✅' if meets_prec else '❌'} (≥{min_requirements['precision']})")
        
        # Check AUC
        auc = model_results['auc_roc']
        meets_auc = auc >= min_requirements['auc_roc']
        print(f"  AUC-ROC: {auc:.3f} {'✅' if meets_auc else '❌'} (≥{min_requirements['auc_roc']})")
        
        overall_sufficient = meets_sens and meets_prec and meets_auc
        
        assessment[model_name] = {
            'sufficient': overall_sufficient,
            'meets_sensitivity': meets_sens,
            'meets_precision': meets_prec,
            'meets_auc': meets_auc,
            'overall_score': (sensitivity + precision + auc) / 3
        }
        
        print(f"  Overall Assessment: {'✅ SUFFICIENT' if overall_sufficient else '❌ NEEDS IMPROVEMENT'}")
    
    # Final recommendation
    print(f"\n🎯 FINAL RECOMMENDATION:")
    print("=" * 40)
    
    best_model = max(assessment.keys(), key=lambda k: assessment[k]['overall_score'])
    best_sufficient = assessment[best_model]['sufficient']
    
    if best_sufficient:
        print(f"✅ MODELS ARE SUFFICIENT!")
        print(f"🏆 Best model: {best_model}")
        print(f"📊 Overall score: {assessment[best_model]['overall_score']:.3f}")
        print("\n💡 Recommendations:")
        print("  - Deploy current models for testing")
        print("  - Monitor performance in real scenarios")
        print("  - Consider fine-tuning thresholds")
        print("  - No need for immediate scaling up")
    else:
        print(f"❌ MODELS NEED IMPROVEMENT")
        print(f"🎯 Areas to improve:")
        
        for model_name, model_assess in assessment.items():
            if not model_assess['sufficient']:
                print(f"\n{model_name}:")
                if not model_assess['meets_sensitivity']:
                    print("  - Improve seizure detection rate (sensitivity)")
                if not model_assess['meets_precision']:
                    print("  - Reduce false alarms (precision)")
                if not model_assess['meets_auc']:
                    print("  - Improve overall discrimination (AUC)")
        
        print("\n💡 Scaling up recommendations:")
        print("  - Use more training data (multi-patient)")
        print("  - Try advanced architectures")
        print("  - Implement data augmentation")
        print("  - Consider ensemble methods")
    
    return assessment, best_sufficient

## 6. Interactive Testing Interface

def interactive_model_test():
    """
    Interactive interface to test the models with different scenarios.
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODEL TESTING")
    print("="*60)
    
    # Load models
    rf_model, scaler, cnn_model = load_trained_models()
    
    if None in [rf_model, scaler, cnn_model]:
        print("❌ Cannot proceed without trained models")
        return
    
    # Load test data (you should have this from your previous training)
    print("\n📊 Loading test data...")
    print("Note: Using the test data from your previous training session")
    
    # You would load your actual test data here
    # For now, we'll create instructions
    
    print("\n🔧 Test Options Available:")
    print("1. 📈 Comprehensive Performance Evaluation")
    print("2. 📊 Visualization Suite")  
    print("3. 🌍 Real-World Performance Simulation")
    print("4. ✅ Model Sufficiency Assessment")
    print("5. 🎯 Custom Threshold Testing")
    
    return rf_model, scaler, cnn_model

## 7. Complete Testing Pipeline

def run_complete_model_testing(X_test, y_test, rf_model, scaler, cnn_model):
    """
    Run the complete testing pipeline.
    """
    print("🚀 Starting Complete Model Testing Pipeline...")
    
    # 1. Get predictions
    print("\n1️⃣ Generating predictions...")
    
    # RF predictions
    test_features = extract_features(X_test)
    test_features_scaled = scaler.transform(test_features)
    y_pred_rf = rf_model.predict(test_features_scaled)
    y_proba_rf = rf_model.predict_proba(test_features_scaled)[:, 1]
    
    # CNN predictions
    X_test_cnn = X_test.transpose(0, 2, 1)
    y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int).flatten()
    y_proba_cnn = cnn_model.predict(X_test_cnn).flatten()
    
    # 2. Comprehensive evaluation
    print("\n2️⃣ Running comprehensive evaluation...")
    results = comprehensive_evaluation(y_test, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn)
    
    # 3. Create visualizations
    print("\n3️⃣ Creating visualizations...")
    create_comprehensive_visualizations(y_test, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn)
    
    # 4. Real-world simulation
    print("\n4️⃣ Running real-world simulation...")
    threshold_results, opt_rf_thresh, opt_cnn_thresh = simulate_real_world_performance(
        rf_model, scaler, cnn_model, X_test, y_test
    )
    
    # 5. Sufficiency assessment
    print("\n5️⃣ Assessing model sufficiency...")
    assessment, is_sufficient = assess_model_sufficiency(results, threshold_results)
    
    return results, assessment, is_sufficient

## 8. Easy Integration with Your Existing Notebook

def test_current_models_integration():
    """
    Easy integration function that works with your existing notebook.
    Run this after your training is complete.
    """
    print("🔄 Loading your trained models and test data...")
    
    try:
        # Load the models you just trained
        rf_model = joblib.load('/content/new/rf_seizure_model.joblib')
        scaler = joblib.load('/content/new/feature_scaler.joblib')
        cnn_model = keras.models.load_model('/content/new/cnn_seizure_model.h5')
        print("✅ Models loaded successfully")
        
        # Check if test data exists in global scope
        try:
            # Try to access the variables from your training notebook
            test_windows = globals().get('X_test_raw', None)
            test_labels = globals().get('y_test', None)
            test_features_scaled = globals().get('X_test_feat_scaled', None)
            
            if test_windows is None or test_labels is None:
                print("⚠️ Test data not found in current session.")
                print("💡 Please run this after your training notebook or recreate test data.")
                return create_new_test_data()
            
            print(f"📊 Found test data: {len(test_labels)} samples")
            
            # Run comprehensive testing
            return run_testing_with_existing_data(
                rf_model, scaler, cnn_model, 
                test_windows, test_labels, test_features_scaled
            )
            
        except Exception as e:
            print(f"⚠️ Error accessing test data: {e}")
            return create_new_test_data()
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Please ensure your models are trained and saved first.")
        return None

def create_new_test_data():
    """
    Create new test data from the saved processed data.
    """
    print("\n🔄 Creating new test data from your CHB-MIT dataset...")
    
    try:
        # Recreate the data processing pipeline
        DATA_PATH = '/content/physionet.org/files/chbmit/1.0.0/'
        
        # Load data again (just for testing)
        raw_data_segments, raw_labels_segments, file_info = load_chb_mit_data(
            DATA_PATH, patient_id='chb01', max_files=3
        )
        
        if raw_data_segments is None:
            print("❌ Cannot load CHB-MIT data for testing")
            return None
        
        # Preprocess
        processed_data_segments = preprocess_eeg_segments(raw_data_segments, 256)
        
        # Create windows
        window_size_samples = 5 * 256
        overlap_samples = int(0.5 * window_size_samples)
        
        X_windows, y_windows = create_windows_from_continuous_data(
            processed_data_segments, raw_labels_segments, window_size_samples, overlap_samples
        )
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_windows, y_windows, test_size=0.2, random_state=42, stratify=y_windows
        )
        
        print(f"✅ Created test data: {len(y_test)} samples")
        print(f"   Seizure samples: {np.sum(y_test)}")
        print(f"   Normal samples: {len(y_test) - np.sum(y_test)}")
        
        # Now run testing
        rf_model = joblib.load('/content/new/rf_seizure_model.joblib')
        scaler = joblib.load('/content/new/feature_scaler.joblib')
        cnn_model = keras.models.load_model('/content/new/cnn_seizure_model.h5')
        
        return run_testing_with_existing_data(rf_model, scaler, cnn_model, X_test, y_test, None)
        
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        return None

def run_testing_with_existing_data(rf_model, scaler, cnn_model, X_test, y_test, X_test_feat_scaled=None):
    """
    Run testing with the provided data.
    """
    print("\n🚀 Starting comprehensive model testing...")
    
    try:
        # Generate features if not provided
        if X_test_feat_scaled is None:
            print("🔧 Extracting features for Random Forest...")
            test_features = extract_features(X_test)
            X_test_feat_scaled = scaler.transform(test_features)
        
        # Get RF predictions
        print("🔮 Getting Random Forest predictions...")
        y_pred_rf = rf_model.predict(X_test_feat_scaled)
        y_proba_rf = rf_model.predict_proba(X_test_feat_scaled)[:, 1] if rf_model.predict_proba(X_test_feat_scaled).shape[1] > 1 else rf_model.predict_proba(X_test_feat_scaled)[:, 0]
        
        # Get CNN predictions
        print("🔮 Getting CNN predictions...")
        X_test_cnn = X_test.transpose(0, 2, 1)
        y_pred_cnn = (cnn_model.predict(X_test_cnn, verbose=0) > 0.5).astype(int).flatten()
        y_proba_cnn = cnn_model.predict(X_test_cnn, verbose=0).flatten()
        
        # Run comprehensive evaluation
        print("\n📊 Running comprehensive evaluation...")
        results = comprehensive_evaluation(y_test, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn)
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        create_comprehensive_visualizations(y_test, y_pred_rf, y_pred_cnn, y_proba_rf, y_proba_cnn)
        
        # Run real-world simulation
        print("\n🌍 Running real-world simulation...")
        threshold_results, opt_rf_thresh, opt_cnn_thresh = simulate_real_world_performance(
            rf_model, scaler, cnn_model, X_test, y_test
        )
        
        # Assess sufficiency
        print("\n✅ Assessing model sufficiency...")
        assessment, is_sufficient = assess_model_sufficiency(results, threshold_results)
        
        return results, assessment, is_sufficient
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

## 9. Simple One-Command Testing

def quick_model_test():
    """
    One-command testing function - just run this!
    """
    print("🎯 QUICK MODEL TESTING")
    print("=" * 40)
    print("Testing your current CHB-MIT seizure classification models...")
    
    return test_current_models_integration()

## 10. Usage Instructions

print("🎯 MODEL TESTING SUITE READY!")
print("=" * 50)
print("\n🚀 EASY TESTING - Just run one of these:")
print("\n# Option 1: Quick Test (Recommended)")
print("results, assessment, is_sufficient = quick_model_test()")
print("\n# Option 2: Integrated Test")  
print("results, assessment, is_sufficient = test_current_models_integration()")
print("\n✨ This will automatically:")
print("  • Load your trained models")
print("  • Recreate test data if needed") 
print("  • Run comprehensive evaluation")
print("  • Show visualizations")
print("  • Give clear recommendations")
print("\n🎯 The result will tell you:")
print("✅ Whether your models are sufficient for deployment")
print("📊 Detailed performance breakdown")
print("💡 Whether to scale up or deploy as-is")
