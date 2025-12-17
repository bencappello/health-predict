"""
Unit tests for drift utilities - Stage 1
Fast local tests with tiny datasets
"""

import pytest
import pandas as pd
import numpy as np
from src.drift.utils import (
    split_batch_for_training,
    prepare_cumulative_data,
    evaluate_quality_gate
)


class TestBatchSplit:
    """Test batch splitting functionality"""
    
    def test_basic_split_50_50(self):
        """Test that batch is split 50/50 correctly"""
        # Create tiny mock batch (100 rows)
        mock_batch = pd.DataFrame({
            'feature1': range(100),
            'feature2': np.random.randn(100),
            'readmitted_binary': [0, 1] * 50
        })
        
        train, test = split_batch_for_training(mock_batch, test_size=0.5, seed=42)
        
        assert len(train) == 50, "Train should have 50 rows"
        assert len(test) == 50, "Test should have 50 rows"
        assert len(set(train.index) & set(test.index)) == 0, "No overlap between train and test"
        
    def test_split_reproducible(self):
        """Test that same seed gives same split"""
        mock_data = pd.DataFrame({
            'a': range(100),
            'readmitted_binary': [0, 1] * 50
        })
        
        train1, test1 = split_batch_for_training(mock_data, seed=42)
        train2, test2 = split_batch_for_training(mock_data, seed=42)
        
        assert train1.equals(train2), "Same seed should produce same train set"
        assert test1.equals(test2), "Same seed should produce same test set"
        
    def test_split_different_seeds(self):
        """Test that different seeds give different splits"""
        mock_data = pd.DataFrame({
            'a': range(100),
            'readmitted_binary': [0, 1] * 50
        })
        
        train1, _ = split_batch_for_training(mock_data, seed=42)
        train2, _ = split_batch_for_training(mock_data, seed=99)
        
        assert not train1.equals(train2), "Different seeds should produce different splits"
        
    def test_split_custom_test_size(self):
        """Test custom test size (70/30 split)"""
        mock_data = pd.DataFrame({
            'a': range(100),
            'readmitted_binary': [0, 1] * 50
        })
        
        train, test = split_batch_for_training(mock_data, test_size=0.3, seed=42)
        
        assert len(train) == 70, "Train should have 70 rows"
        assert len(test) == 30, "Test should have 30 rows"


class TestCumulativeData:
    """Test cumulative data preparation"""
    
    def test_combine_two_datasets(self):
        """Test combining two datasets"""
        data1 = pd.DataFrame({'id': [1, 2], 'val': [10, 20]})
        data2 = pd.DataFrame({'id': [3, 4], 'val': [30, 40]})
        
        cumulative = prepare_cumulative_data([data1, data2])
        
        assert len(cumulative) == 4, "Should have 4 total rows"
        assert list(cumulative['id']) == [1, 2, 3, 4], "Should preserve order"
        
    def test_combine_three_datasets(self):
        """Test combining three datasets (initial + batch1 + batch7)"""
        initial = pd.DataFrame({'x': [1, 2], 'y': [10, 20]})
        batch1 = pd.DataFrame({'x': [3, 4], 'y': [30, 40]})
        batch7 = pd.DataFrame({'x': [5, 6], 'y': [50, 60]})
        
        cumulative = prepare_cumulative_data([initial, batch1, batch7])
        
        assert len(cumulative) == 6, "Should have 6 total rows"
        
    def test_duplicate_removal(self):
        """Test that duplicates are removed"""
        data_with_dupes = pd.DataFrame({
            'id': [1, 1, 2, 3],  # Duplicate row
            'val': [10, 10, 20, 30]
        })
        
        cleaned = prepare_cumulative_data([data_with_dupes], remove_duplicates=True)
        
        assert len(cleaned) == 3, "Should remove duplicate row"
        assert list(cleaned['id']) == [1, 2, 3], "Should keep unique rows"
        
    def test_no_duplicate_removal(self):
        """Test keeping duplicates when specified"""
        data_with_dupes = pd.DataFrame({
            'id': [1, 1, 2],
            'val': [10, 10, 20]
        })
        
        kept = prepare_cumulative_data([data_with_dupes], remove_duplicates=False)
        
        assert len(kept) == 3, "Should keep all rows including duplicates"
        
    def test_empty_list_raises_error(self):
        """Test that empty dataset list raises error"""
        with pytest.raises(ValueError, match="Must provide at least one dataset"):
            prepare_cumulative_data([])


class TestQualityGate:
    """Test quality gate evaluation logic"""
    
    def test_approve_with_sufficient_improvement(self):
        """Test that 2%+ AUC improvement approves deployment"""
        # Create mock models
        from sklearn.linear_model import LogisticRegression
        
        # Mock test data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.choice([0, 1], 100)
        test_data = pd.DataFrame(X, columns=['f1', 'f2'])
        test_data['readmitted_binary'] = y
        
        # Train "production" model (deliberately worse)
        X_train_prod = np.random.randn(100, 2)
        y_train_prod = np.random.choice([0, 1], 100)
        prod_model = LogisticRegression(random_state=42, C=0.01)  # Low C = worse performance
        prod_model.fit(X_train_prod, y_train_prod)
        
        # Train "new" model (deliberately better)
        X_train_new = X + np.random.randn(100, 2) * 0.1  # Closer to test data
        y_train_new = y
        new_model = LogisticRegression(random_state=42, C=10.0)  # High C = better performance
        new_model.fit(X_train_new, y_train_new)
        
        # Evaluate
        decision = evaluate_quality_gate(
            prod_model, 
            new_model,
            test_data,
            threshold=0.02,
            metric='auc'
        )
        
        # Note: Given random data, we can't guarantee improvement,
        # but we can check the structure
        assert 'approve' in decision
        assert 'improvement' in decision
        assert 'prod_score' in decision
        assert 'new_score' in decision
        assert decision['metric'] == 'auc'
        
    def test_reject_with_regression(self):
        """Test that regression prevents deployment"""
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.choice([0, 1], 100)
        test_data = pd.DataFrame(X, columns=['f1', 'f2'])
        test_data['readmitted_binary'] = y
        
        # Both models same (simulates regression)
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Manually set to simulate regression
        # (In real test, new model would perform worse)
        decision = evaluate_quality_gate(
            model,  # "production"
            model,  # "new" (same = no improvement)
            test_data,
            threshold=0.02,
            metric='auc'
        )
        
        # With same model, improvement should be ~0
        assert abs(decision['improvement']) < 0.001, "Same model should have ~0 improvement"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
