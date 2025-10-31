"""
Stage 1 experiment: DICOM to BIDS organization with metrics collection.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Import base classes
from research.common.experiment_base import ExperimentBase
from metrics_stage1 import Stage1MetricsCollector, Stage1PatientMetrics

# Import production Stage 1 script components
# Assuming reorganize_folders.py is in the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from reorganize_folders import DicomScanner, BidsOrganizer, setup_logging

# Import research utilities
from research.common.dataset_sampler import DatasetSampler
from research.common.ground_truth_parser import create_parser


class Stage1Experiment(ExperimentBase):
    """
    Experiment for Stage 1: DICOM to BIDS Organization.
    
    Runs reorganize_folders.py on selected patients and collects metrics.
    """
    
    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 patient_list: List[str],
                 dataset_type: str = "upenn-gbm",
                 action: str = "copy",
                 max_workers: int = None,
                 streaming_mode: bool = False):
        """
        Initialize Stage 1 experiment.
        
        Args:
            input_dir: Input directory with DICOM data
            output_dir: Output directory for BIDS data
            patient_list: List of patient IDs
            dataset_type: Dataset type ('upenn-gbm' or 'ms-dataset')
            action: File action ('copy' or 'move')
            max_workers: Number of parallel workers
            streaming_mode: Enable streaming processing for large datasets
        """
        super().__init__(
            stage_name="stage1",
            input_dir=input_dir,
            output_dir=output_dir,
            patient_list=patient_list,
            dataset_type=dataset_type
        )
        
        self.action = action
        self.max_workers = max_workers
        self.streaming_mode = streaming_mode  # PERFORMANCE FIX: Enable streaming mode
        
        # Initialize metrics collector
        self.metrics_collector = Stage1MetricsCollector(str(self.metrics_dir))
        
        # Initialize ground truth parser
        self.gt_parser = create_parser(dataset_type)

        # Timing and memory tracking for overall experiment
        self.experiment_start_time = None
        self.experiment_mem_start = None
        self.scan_time = 0.0
        self.organize_time = 0.0
        
        # Setup logging for reorganize_folders.py
        log_file = self.output_dir / "logs" / "stage1.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        setup_logging(str(log_file))

    def _merge_studies_by_date(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge studies with same date into single study (for UPENN dataset).
        
        UPENN-GBM has pre-organized structure where multiple studies 
        may exist for same scan date. This merges them into one study
        to create single BIDS session per date.
        
        Args:
            collected_data: Dict of patient_id -> PatientData
            
        Returns:
            Updated collected_data with merged studies
        """
        from collections import defaultdict
        from reorganize_folders import PatientData, StudyInfo
        
        merged_data = {}
        
        for patient_id, patient_data in collected_data.items():
            # Group studies by date (without time)
            studies_by_date = defaultdict(list)
            
            for study_uid, study_info in patient_data.studies.items():
                date_key = study_info.study_datetime.strftime('%Y%m%d')
                studies_by_date[date_key].append((study_uid, study_info))
            
            # Create merged studies
            merged_studies = {}
            
            for date_key, study_list in studies_by_date.items():
                if len(study_list) == 1:
                    # Only one study for this date, keep as is
                    study_uid, study_info = study_list[0]
                    merged_studies[study_uid] = study_info
                else:
                    # Multiple studies for same date - merge them
                    # Use first study as base
                    first_uid, first_study = study_list[0]
                    
                    # Collect all series from all studies of this date
                    merged_series = {}
                    for study_uid, study_info in study_list:
                        merged_series.update(study_info.series)
                    
                    # Create new merged StudyInfo
                    merged_study = StudyInfo(
                        uid=first_uid,  # Keep first UID
                        series=merged_series,
                        study_datetime=first_study.study_datetime
                    )
                    
                    merged_studies[first_uid] = merged_study
            
            # Create updated PatientData
            merged_patient = PatientData(
                original_id=patient_data.original_id,
                studies=merged_studies
            )
            
            merged_data[patient_id] = merged_patient
        
        return merged_data
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run Stage 1 experiment on all patients."""
        print("\n" + "="*70)
        print("Stage 1 Experiment: DICOM to BIDS Organization")
        print("="*70)
        print(f"Processing {len(self.patient_list)} patients")
        print("="*70 + "\n")
        
        # Track overall metrics
        self.experiment_start_time = time.perf_counter()
        self.experiment_mem_start = self.metrics_collector.get_memory_usage_mb()
        
        # Phase 1: Scan all patients
        print("Phase 1: Scanning DICOM files for all patients...")
        scan_start = time.perf_counter()
        all_patient_data = self._scan_all_patients()
        self.scan_time = time.perf_counter() - scan_start
        print(f"Scanning completed in {self.scan_time:.2f} seconds\n")
        
        if not all_patient_data:
            print("No valid patient data found")
            summary = {
                'total_patients': len(self.patient_list),
                'successful_patients': 0,
                'failed_patients': len(self.patient_list)
            }
            return summary
        
        # UPENN preprocessing: merge studies by date
        if self.dataset_type == 'upenn-gbm':
            print("Preprocessing: Merging studies by date for UPENN dataset...")
            all_patient_data = self._merge_studies_by_date(all_patient_data)
            print("Preprocessing completed\n")

        # Calculate actual number of workers
        if self.max_workers:
            actual_workers = self.max_workers
        else:
            actual_workers = min(os.cpu_count(), len(all_patient_data))
            if len(all_patient_data) == 1:
                actual_workers = 1
        
        # Phase 2: Organize all patients in parallel
        print(f"Phase 2: Organizing to BIDS (parallel, max_workers={actual_workers})...")
        organize_start = time.perf_counter()
        mem_before_organize = self.metrics_collector.get_memory_usage_mb()
        
        organizer = BidsOrganizer(
            str(self.output_dir / "rawdata"),
            self.action,
            max_parallel_files=1,
            max_workers=self.max_workers,
            metrics_callback=self._on_patient_processed,
            streaming_mode=self.streaming_mode  # PERFORMANCE FIX: Pass streaming mode
        )
        
        organizer.organize_to_bids(all_patient_data)
        
        self.organize_time = time.perf_counter() - organize_start
        mem_peak = self.metrics_collector.get_memory_usage_mb()
        total_time = time.perf_counter() - self.experiment_start_time
        
        print(f"\nOrganization completed in {self.organize_time:.2f} seconds")
        print(f"Total experiment time: {total_time:.2f} seconds\n")
        
        # Compute average metrics
        num_successful = len([m for m in self.metrics_collector.patient_metrics 
                            if m.processing_status == 'success'])
        
        if num_successful > 0:
            mean_total_time = total_time / num_successful
            mean_scan_time = self.scan_time / len(self.patient_list)
            mean_organize_time = self.organize_time / num_successful
        else:
            mean_total_time = mean_scan_time = mean_organize_time = 0.0
        
        # Save all metrics
        self.metrics_collector.save_all_metrics()
        self.save_metadata()
        
        # Print and save summary
        summary = self.metrics_collector.compute_summary()
        
        # Add average metrics
        summary['average_metrics'] = {
            'mean_total_time_sec': mean_total_time,
            'mean_scan_time_sec': mean_scan_time,
            'mean_organize_time_sec': mean_organize_time,
            'peak_memory_mb': mem_peak,
            'num_workers': actual_workers,
            'total_experiment_time_sec': total_time,
            'note': 'Averaged across all patients in parallel batch'
        }
        
        self.print_summary(summary)
        self.save_results(summary, "experiment_summary.json")
        
        return summary
    
    def _scan_all_patients(self) -> Dict[str, Any]:
        """
        Scan DICOM files for all patients sequentially.
        
        Returns:
            Dict of patient_id -> PatientData
        """
        scanner = DicomScanner()
        all_patient_data = {}
        
        for idx, patient_id in enumerate(self.patient_list, 1):
            self.print_progress(idx, len(self.patient_list), patient_id)
            
            patient_path = self.get_patient_path(patient_id)
            
            try:
                patient_data = scanner.scan_directory(str(patient_path))
                if patient_data:
                    all_patient_data[patient_id] = list(patient_data.values())[0]
                    print(f"  ✓ Scanned successfully")
                else:
                    print(f"  ⚠ No DICOM data found")
            except Exception as e:
                print(f"  ✗ Failed to scan: {e}")
        
        print(f"\nSuccessfully scanned {len(all_patient_data)} patients")
        return all_patient_data
    
    def _on_patient_processed(self, patient_id: str, callback_data: Dict[str, Any]):
        """
        Callback called by BidsOrganizer after processing each patient.
        Collects metrics from the callback data.
        
        Args:
            patient_id: Patient identifier
            callback_data: Dict with processing results from BidsOrganizer
        """
        # Get ground truth modalities
        original_id = callback_data.get('original_patient_id', patient_id)
        patient_path = self.get_patient_path(original_id)
        gt_modalities = self.gt_parser.parse_patient_modalities(str(patient_path))
        
        # Extract detected modalities from callback
        detected = callback_data.get('detected_modalities', [])
        
        # Calculate accuracy
        if gt_modalities:
            correct = len([m for m in detected if m in gt_modalities])
            accuracy = correct / len(gt_modalities)
        else:
            accuracy = 0.0
        
        # Create metrics
        metrics = Stage1PatientMetrics(
            patient_id=patient_id,
            processing_status=callback_data.get('processing_status', 'unknown'),
            processing_time_sec=0.0,  # Not available per-patient in parallel mode
            memory_start_mb=0.0,      # Not available per-patient
            memory_peak_mb=0.0,       # Not available per-patient
            memory_delta_mb=0.0,      # Not available per-patient
            detected_modalities=sorted(detected),
            ground_truth_modalities=gt_modalities,
            missing_modalities=[m for m in gt_modalities if m not in detected],
            modality_accuracy=accuracy,
            num_series_organized=callback_data.get('num_series_organized', 0),
            num_series_total=0,       # Not available in callback
            scan_phase_sec=0.0,       # Not available per-patient
            organize_phase_sec=0.0,   # Not available per-patient
            failure_reason=callback_data.get('failure_reason')
        )
        
        # Add to collector (thread-safe)
        self.metrics_collector.add_patient_metrics(metrics)
        
        # Print progress
        print(f"  ✓ Collected metrics for {patient_id}: "
            f"{len(detected)}/{len(gt_modalities)} modalities, "
            f"accuracy={accuracy:.2%}")


def main():
    """CLI for Stage 1 experiment."""
    parser = argparse.ArgumentParser(
        description="Stage 1 Experiment: DICOM to BIDS with metrics"
    )
    
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='Path to dataset (e.g., UPENN-GBM)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for results and metrics'
    )
    
    parser.add_argument(
        '--dataset-type',
        choices=['upenn-gbm', 'ms-dataset'],
        default='upenn-gbm',
        help='Type of dataset'
    )
    
    parser.add_argument(
        '--num-patients',
        type=int,
        default=None,
        help='Number of patients to process (default: all)'
    )
    
    parser.add_argument(
        '--sampling-strategy',
        choices=['random', 'first', 'last'],
        default='first',
        help='Patient sampling strategy'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--action',
        choices=['copy', 'move'],
        default='copy',
        help='File action (copy or move)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--streaming-mode',
        action='store_true',
        help='Enable streaming mode for large datasets (recommended for 50+ patients)'
    )
    
    args = parser.parse_args()
    
    # Sample patients
    sampler = DatasetSampler(
        dataset_path=args.dataset_path,
        random_seed=args.seed
    )
    
    if args.num_patients:
        patient_list = sampler.sample_patients(
            n=args.num_patients,
            mode=args.sampling_strategy
        )
    else:
        patient_list = sampler.get_all_patients()
    
    print(f"Selected {len(patient_list)} patients")
    
    # Create experiment
    experiment = Stage1Experiment(
        input_dir=args.dataset_path,
        output_dir=args.output_dir,
        patient_list=patient_list,
        dataset_type=args.dataset_type,
        action=args.action,
        max_workers=args.max_workers,
        streaming_mode=args.streaming_mode  # PERFORMANCE FIX: Pass streaming mode
    )
    
    # Run experiment
    summary = experiment.run_experiment()
    
    # Exit code based on results
    if summary['failed_patients'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()