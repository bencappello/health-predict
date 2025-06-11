#!/usr/bin/env python3
"""
Batch Processing Simulation for Health Predict Drift Monitoring

This script implements realistic continuous data processing simulation with:
- Time-based batch processing simulation
- Realistic data arrival patterns and timing
- Batch size variation and realistic data scenarios
- Processing backlog and catch-up mechanisms
- Batch processing performance monitoring

Author: Health Predict ML Engineering Team
Created: 2025-01-24 (Week 5 - Step 18)
"""

import argparse
import boto3
import pandas as pd
import numpy as np
import logging
import time
import json
import uuid
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BatchArrivalPattern:
    """Configuration for realistic batch arrival patterns"""
    pattern_name: str
    base_interval_hours: float
    variation_factor: float  # 0.0-1.0, how much to vary from base interval
    size_min: int
    size_max: int
    peak_hours: List[int]  # Hours of day with higher arrival rates
    weekend_factor: float  # Factor to apply on weekends (0.5 = half rate)

@dataclass
class BatchMetrics:
    """Metrics for batch processing performance"""
    batch_id: str
    arrival_timestamp: str
    processing_start: str
    processing_end: str
    processing_duration_seconds: float
    batch_size: int
    data_quality_score: float
    arrival_pattern: str
    s3_path: str
    error_count: int = 0
    status: str = "completed"

class BatchProcessingSimulator:
    """
    Comprehensive batch processing simulator for realistic data scenarios
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the batch processing simulator"""
        self.config = config
        self.s3_client = boto3.client('s3')
        self.bucket_name = config['S3_BUCKET_NAME']
        
        # Predefined arrival patterns for different scenarios
        self.arrival_patterns = {
            'healthcare_steady': BatchArrivalPattern(
                pattern_name='healthcare_steady',
                base_interval_hours=4.0,  # Every 4 hours base
                variation_factor=0.3,  # 30% variation
                size_min=800,
                size_max=1200,
                peak_hours=[8, 14, 20],  # 8am, 2pm, 8pm
                weekend_factor=0.6  # Reduced weekend activity
            ),
            
            'healthcare_surge': BatchArrivalPattern(
                pattern_name='healthcare_surge',
                base_interval_hours=1.5,  # Frequent arrivals
                variation_factor=0.5,  # High variation
                size_min=1200,
                size_max=2000,
                peak_hours=[9, 10, 11, 15, 16, 17],  # Business hours surge
                weekend_factor=0.4  # Significant weekend reduction
            ),
            
            'healthcare_emergency': BatchArrivalPattern(
                pattern_name='healthcare_emergency',
                base_interval_hours=0.5,  # Very frequent
                variation_factor=0.8,  # Very high variation
                size_min=500,
                size_max=3000,
                peak_hours=[10, 11, 12, 13, 14, 15, 16, 17, 18],  # Extended peak
                weekend_factor=0.9  # Emergency scenarios don't slow on weekends
            ),
            
            'healthcare_maintenance': BatchArrivalPattern(
                pattern_name='healthcare_maintenance',
                base_interval_hours=8.0,  # Slower processing
                variation_factor=0.2,  # Low variation
                size_min=600,
                size_max=1000,
                peak_hours=[22, 23, 0, 1, 2, 3],  # Maintenance hours
                weekend_factor=1.2  # More maintenance on weekends
            )
        }
        
        self.processing_metrics: List[BatchMetrics] = []
        self.source_data_cache: Optional[pd.DataFrame] = None
        
    def load_source_data(self) -> pd.DataFrame:
        """Load and cache the source data for batch generation"""
        if self.source_data_cache is not None:
            return self.source_data_cache
            
        logger.info("Loading source data for batch simulation...")
        
        try:
            # Download future_data.csv
            future_data_key = 'processed_data/future_data.csv'
            local_temp_file = f'/tmp/future_data_cache_{uuid.uuid4().hex[:8]}.csv'
            
            self.s3_client.download_file(
                self.bucket_name, 
                future_data_key, 
                local_temp_file
            )
            
            # Load and cache the data
            self.source_data_cache = pd.read_csv(local_temp_file)
            os.remove(local_temp_file)
            
            logger.info(f"Loaded source data with {len(self.source_data_cache)} rows")
            return self.source_data_cache
            
        except Exception as e:
            logger.error(f"Error loading source data: {str(e)}")
            raise
    
    def calculate_batch_arrival_time(
        self, 
        pattern: BatchArrivalPattern, 
        current_time: datetime
    ) -> datetime:
        """Calculate the next batch arrival time based on realistic patterns"""
        
        # Base interval with variation
        base_minutes = pattern.base_interval_hours * 60
        variation = random.uniform(-pattern.variation_factor, pattern.variation_factor)
        actual_minutes = base_minutes * (1 + variation)
        
        # Apply peak hour adjustments
        current_hour = current_time.hour
        if current_hour in pattern.peak_hours:
            # Increase frequency during peak hours
            actual_minutes *= 0.7  # 30% faster arrival
        
        # Apply weekend factor
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            actual_minutes *= (1 / pattern.weekend_factor)
        
        # Ensure minimum interval of 15 minutes
        actual_minutes = max(actual_minutes, 15)
        
        next_arrival = current_time + timedelta(minutes=actual_minutes)
        return next_arrival
    
    def generate_realistic_batch(
        self, 
        pattern: BatchArrivalPattern,
        batch_id: str,
        arrival_time: datetime
    ) -> Dict[str, Any]:
        """Generate a realistic batch with data quality variations"""
        
        start_time = time.time()
        source_data = self.load_source_data()
        
        # Determine batch size with realistic variation
        batch_size = random.randint(pattern.size_min, pattern.size_max)
        
        # Calculate start position (with some overlap to simulate real data)
        max_start = max(0, len(source_data) - batch_size)
        start_row = random.randint(0, max_start) if max_start > 0 else 0
        end_row = min(start_row + batch_size, len(source_data))
        
        # Extract batch data
        batch_data = source_data.iloc[start_row:end_row].copy()
        
        # Apply realistic data quality issues based on time patterns
        data_quality_score = self.apply_data_quality_variations(
            batch_data, pattern, arrival_time
        )
        
        # Save batch to local file
        local_batch_path = f'/tmp/{batch_id}.csv'
        batch_data.to_csv(local_batch_path, index=False)
        
        # Upload to S3
        s3_batch_key = f"{self.config['DRIFT_BATCH_DATA_S3_PREFIX']}/{batch_id}.csv"
        
        try:
            self.s3_client.upload_file(
                local_batch_path, 
                self.bucket_name, 
                s3_batch_key
            )
            
            # Cleanup
            os.remove(local_batch_path)
            
            processing_time = time.time() - start_time
            
            # Create batch metadata
            batch_info = {
                'batch_id': batch_id,
                'arrival_timestamp': arrival_time.isoformat(),
                'processing_timestamp': datetime.now().isoformat(),
                'batch_size': len(batch_data),
                'start_row': start_row,
                'end_row': end_row,
                'data_quality_score': data_quality_score,
                'arrival_pattern': pattern.pattern_name,
                's3_path': s3_batch_key,
                'processing_duration_seconds': processing_time,
                'status': 'completed'
            }
            
            logger.info(f"Generated batch {batch_id}: {len(batch_data)} rows, "
                       f"quality={data_quality_score:.3f}, pattern={pattern.pattern_name}")
            
            return batch_info
            
        except Exception as e:
            logger.error(f"Error generating batch {batch_id}: {str(e)}")
            if os.path.exists(local_batch_path):
                os.remove(local_batch_path)
            raise
    
    def apply_data_quality_variations(
        self, 
        batch_data: pd.DataFrame, 
        pattern: BatchArrivalPattern,
        arrival_time: datetime
    ) -> float:
        """Apply realistic data quality variations based on time patterns"""
        
        # Base quality score starts high
        quality_score = 1.0
        
        # Time-based quality variations
        hour = arrival_time.hour
        
        # Quality tends to be lower during shift changes and peak hours
        if hour in [7, 15, 23]:  # Shift change hours
            if random.random() < 0.4:  # 40% chance of quality issues
                quality_impact = random.uniform(0.05, 0.15)
                quality_score -= quality_impact
                self._introduce_missing_values(batch_data, quality_impact)
        
        # Weekend data might have different quality patterns
        if arrival_time.weekday() >= 5:  # Weekend
            if random.random() < 0.3:  # 30% chance of quality issues
                quality_impact = random.uniform(0.02, 0.08)
                quality_score -= quality_impact
                self._introduce_data_inconsistencies(batch_data, quality_impact)
        
        # Emergency/surge patterns have more quality issues
        if pattern.pattern_name == 'healthcare_emergency':
            if random.random() < 0.6:  # 60% chance during emergencies
                quality_impact = random.uniform(0.10, 0.25)
                quality_score -= quality_impact
                self._introduce_multiple_quality_issues(batch_data, quality_impact)
        
        # Maintenance patterns have better quality but different timing
        elif pattern.pattern_name == 'healthcare_maintenance':
            if random.random() < 0.1:  # Only 10% chance of issues
                quality_impact = random.uniform(0.01, 0.05)
                quality_score -= quality_impact
        
        return max(quality_score, 0.0)  # Ensure non-negative
    
    def _introduce_missing_values(self, batch_data: pd.DataFrame, intensity: float):
        """Introduce missing values to simulate data quality issues"""
        n_missing = int(len(batch_data) * intensity)
        if n_missing > 0:
            missing_indices = random.sample(range(len(batch_data)), n_missing)
            missing_columns = random.sample(
                list(batch_data.columns), 
                min(3, len(batch_data.columns))
            )
            
            for idx in missing_indices:
                for col in missing_columns:
                    if random.random() < 0.4:  # 40% chance per column
                        batch_data.iloc[idx, batch_data.columns.get_loc(col)] = None
    
    def _introduce_data_inconsistencies(self, batch_data: pd.DataFrame, intensity: float):
        """Introduce data inconsistencies to simulate real-world issues"""
        n_inconsistent = int(len(batch_data) * intensity * 0.5)  # Less frequent than missing
        
        if n_inconsistent > 0:
            inconsistent_indices = random.sample(range(len(batch_data)), n_inconsistent)
            
            for idx in inconsistent_indices:
                # Randomly modify some categorical values to simulate entry errors
                categorical_columns = batch_data.select_dtypes(include=['object']).columns
                if len(categorical_columns) > 0:
                    col = random.choice(categorical_columns)
                    if random.random() < 0.3:  # 30% chance
                        # Introduce a typo or inconsistency
                        original_value = str(batch_data.iloc[idx, batch_data.columns.get_loc(col)])
                        if original_value and len(original_value) > 1:
                            # Simple character substitution
                            pos = random.randint(0, len(original_value) - 1)
                            new_value = original_value[:pos] + 'X' + original_value[pos+1:]
                            batch_data.iloc[idx, batch_data.columns.get_loc(col)] = new_value
    
    def _introduce_multiple_quality_issues(self, batch_data: pd.DataFrame, intensity: float):
        """Introduce multiple types of quality issues for emergency scenarios"""
        # Combine missing values and inconsistencies
        self._introduce_missing_values(batch_data, intensity * 0.6)
        self._introduce_data_inconsistencies(batch_data, intensity * 0.4)
        
        # Additional emergency-specific issues
        n_extreme = int(len(batch_data) * intensity * 0.1)
        if n_extreme > 0:
            extreme_indices = random.sample(range(len(batch_data)), n_extreme)
            numeric_columns = batch_data.select_dtypes(include=[np.number]).columns
            
            for idx in extreme_indices:
                if len(numeric_columns) > 0:
                    col = random.choice(numeric_columns)
                    # Introduce extreme outliers
                    if random.random() < 0.2:  # 20% chance
                        batch_data.iloc[idx, batch_data.columns.get_loc(col)] = 999999
    
    def simulate_processing_backlog(self) -> Dict[str, Any]:
        """Simulate processing backlog and catch-up mechanisms"""
        logger.info("Analyzing processing backlog...")
        
        try:
            # Check current batch backlog
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.config['DRIFT_BATCH_DATA_S3_PREFIX']}/",
                MaxKeys=self.config.get('BATCH_PROCESSING_BACKLOG_LIMIT', 10) + 5
            )
            
            current_batches = response.get('Contents', [])
            backlog_count = len(current_batches)
            
            # Calculate backlog metrics
            if current_batches:
                oldest_batch = min(current_batches, key=lambda x: x['LastModified'])
                newest_batch = max(current_batches, key=lambda x: x['LastModified'])
                
                backlog_age_hours = (
                    datetime.now().replace(tzinfo=oldest_batch['LastModified'].tzinfo) - 
                    oldest_batch['LastModified']
                ).total_seconds() / 3600
                
                total_backlog_size = sum(batch['Size'] for batch in current_batches)
            else:
                backlog_age_hours = 0
                total_backlog_size = 0
            
            # Determine processing priority adjustments
            if backlog_count > self.config.get('BATCH_PROCESSING_BACKLOG_LIMIT', 10):
                priority_adjustment = 'catch_up_mode'
                recommended_interval = 'reduce_by_50_percent'
            elif backlog_age_hours > 24:
                priority_adjustment = 'urgent_processing'
                recommended_interval = 'reduce_by_30_percent'
            elif backlog_count < 3:
                priority_adjustment = 'normal_processing'
                recommended_interval = 'standard'
            else:
                priority_adjustment = 'elevated_processing'
                recommended_interval = 'reduce_by_20_percent'
            
            backlog_status = {
                'backlog_count': backlog_count,
                'oldest_batch_age_hours': backlog_age_hours,
                'total_backlog_size_bytes': total_backlog_size,
                'priority_adjustment': priority_adjustment,
                'recommended_interval': recommended_interval,
                'status': 'healthy' if backlog_count <= self.config.get('BATCH_PROCESSING_BACKLOG_LIMIT', 10) else 'backlog_alert',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Backlog analysis: {backlog_count} batches, "
                       f"oldest: {backlog_age_hours:.1f}h, "
                       f"status: {backlog_status['status']}")
            
            return backlog_status
            
        except Exception as e:
            logger.error(f"Error analyzing processing backlog: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def run_simulation_cycle(
        self, 
        pattern_name: str = 'healthcare_steady',
        duration_hours: float = 24.0,
        max_batches: int = 50
    ) -> List[Dict[str, Any]]:
        """Run a complete simulation cycle with specified pattern"""
        
        logger.info(f"Starting batch processing simulation: pattern={pattern_name}, "
                   f"duration={duration_hours}h, max_batches={max_batches}")
        
        if pattern_name not in self.arrival_patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}. "
                           f"Available: {list(self.arrival_patterns.keys())}")
        
        pattern = self.arrival_patterns[pattern_name]
        simulation_results = []
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        current_time = start_time
        batch_count = 0
        
        try:
            while current_time < end_time and batch_count < max_batches:
                # Generate batch ID
                batch_id = f"sim_{pattern_name}_{current_time.strftime('%Y%m%d_%H%M%S')}_{batch_count:03d}"
                
                # Generate batch
                batch_info = self.generate_realistic_batch(pattern, batch_id, current_time)
                simulation_results.append(batch_info)
                
                # Analyze backlog periodically
                if batch_count % 5 == 0:  # Every 5 batches
                    backlog_status = self.simulate_processing_backlog()
                    
                    # Adjust timing based on backlog
                    if backlog_status['priority_adjustment'] == 'catch_up_mode':
                        # Reduce arrival frequency to allow catch-up
                        pattern.base_interval_hours *= 1.5
                        logger.info("Backlog detected - reducing arrival frequency")
                    elif backlog_status['priority_adjustment'] == 'normal_processing' and batch_count > 10:
                        # Reset to normal frequency if backlog cleared
                        pattern = self.arrival_patterns[pattern_name]  # Reset to original
                
                # Calculate next arrival time
                next_arrival = self.calculate_batch_arrival_time(pattern, current_time)
                
                # For simulation, we don't actually wait - just advance time
                current_time = next_arrival
                batch_count += 1
                
                # Small delay for realistic logging
                time.sleep(0.1)
            
            simulation_summary = {
                'pattern_name': pattern_name,
                'duration_hours': duration_hours,
                'batches_generated': len(simulation_results),
                'simulation_start': start_time.isoformat(),
                'simulation_end': datetime.now().isoformat(),
                'total_processing_time': sum(b.get('processing_duration_seconds', 0) for b in simulation_results),
                'average_batch_size': sum(b.get('batch_size', 0) for b in simulation_results) / len(simulation_results) if simulation_results else 0,
                'average_quality_score': sum(b.get('data_quality_score', 1.0) for b in simulation_results) / len(simulation_results) if simulation_results else 1.0,
                'final_backlog_status': self.simulate_processing_backlog()
            }
            
            logger.info(f"Simulation complete: {len(simulation_results)} batches generated")
            logger.info(f"Average batch size: {simulation_summary['average_batch_size']:.0f}")
            logger.info(f"Average quality score: {simulation_summary['average_quality_score']:.3f}")
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error in simulation cycle: {str(e)}")
            raise
    
    def cleanup_old_batches(self, keep_days: int = 7) -> Dict[str, Any]:
        """Cleanup old batch files to prevent storage bloat"""
        logger.info(f"Cleaning up batches older than {keep_days} days...")
        
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            # List all batch files
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.config['DRIFT_BATCH_DATA_S3_PREFIX']}/"
            )
            
            old_batches = []
            total_size_cleaned = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        old_batches.append(obj['Key'])
                        total_size_cleaned += obj['Size']
            
            # Delete old batches
            if old_batches:
                for batch_key in old_batches:
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=batch_key)
                
                cleanup_result = {
                    'batches_deleted': len(old_batches),
                    'total_size_cleaned_bytes': total_size_cleaned,
                    'cleanup_timestamp': datetime.now().isoformat(),
                    'cutoff_date': cutoff_date.isoformat()
                }
                
                logger.info(f"Cleanup complete: {len(old_batches)} batches deleted, "
                           f"{total_size_cleaned / 1024 / 1024:.2f} MB freed")
            else:
                cleanup_result = {
                    'batches_deleted': 0,
                    'total_size_cleaned_bytes': 0,
                    'cleanup_timestamp': datetime.now().isoformat(),
                    'message': 'No old batches found for cleanup'
                }
                
                logger.info("No old batches found for cleanup")
            
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Error during batch cleanup: {str(e)}")
            return {
                'error': str(e),
                'cleanup_timestamp': datetime.now().isoformat()
            }

def main():
    """Main function for batch processing simulation"""
    parser = argparse.ArgumentParser(description='Batch Processing Simulation for Health Predict')
    
    parser.add_argument('--pattern', 
                       choices=['healthcare_steady', 'healthcare_surge', 'healthcare_emergency', 'healthcare_maintenance'],
                       default='healthcare_steady',
                       help='Batch arrival pattern to simulate')
    
    parser.add_argument('--duration', type=float, default=24.0,
                       help='Simulation duration in hours')
    
    parser.add_argument('--max-batches', type=int, default=50,
                       help='Maximum number of batches to generate')
    
    parser.add_argument('--cleanup-days', type=int, default=7,
                       help='Days of batches to keep (cleanup older)')
    
    parser.add_argument('--analyze-backlog', action='store_true',
                       help='Analyze current backlog status only')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration from environment variables
    config = {
        'S3_BUCKET_NAME': os.getenv('S3_BUCKET_NAME', 'health-predict-mlops-f9ac6509'),
        'DRIFT_BATCH_DATA_S3_PREFIX': os.getenv('DRIFT_BATCH_DATA_S3_PREFIX', 'drift_monitoring/batch_data'),
        'BATCH_PROCESSING_BACKLOG_LIMIT': int(os.getenv('BATCH_PROCESSING_BACKLOG_LIMIT', '10'))
    }
    
    try:
        # Initialize simulator
        simulator = BatchProcessingSimulator(config)
        
        if args.analyze_backlog:
            # Just analyze current backlog
            backlog_status = simulator.simulate_processing_backlog()
            print(json.dumps(backlog_status, indent=2))
        else:
            # Run full simulation
            results = simulator.run_simulation_cycle(
                pattern_name=args.pattern,
                duration_hours=args.duration,
                max_batches=args.max_batches
            )
            
            # Cleanup old batches
            cleanup_result = simulator.cleanup_old_batches(keep_days=args.cleanup_days)
            
            # Output summary
            summary = {
                'simulation_pattern': args.pattern,
                'batches_generated': len(results),
                'cleanup_result': cleanup_result,
                'final_backlog_status': simulator.simulate_processing_backlog()
            }
            
            print(json.dumps(summary, indent=2))
            
            logger.info("Batch processing simulation completed successfully")
    
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 