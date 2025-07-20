"""
Unit tests for data_loader module
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append('../src')

from src.data_loader import SportsDataLoader


class TestSportsDataLoader(unittest.TestCase):
    """
    Test cases for SportsDataLoader class
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = SportsDataLoader(
            data_dir=self.temp_dir,
            img_size=(224, 224),
            batch_size=32
        )
    
    def tearDown(self):
        """
        Clean up test fixtures
        """
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """
        Test SportsDataLoader initialization
        """
        self.assertEqual(self.data_loader.data_dir, self.temp_dir)
        self.assertEqual(self.data_loader.img_size, (224, 224))
        self.assertEqual(self.data_loader.batch_size, 32)
        self.assertIsNone(self.data_loader.train_df)
        self.assertIsNone(self.data_loader.val_df)
        self.assertIsNone(self.data_loader.test_df)
    
    def test_init_custom_params(self):
        """
        Test SportsDataLoader initialization with custom parameters
        """
        loader = SportsDataLoader(
            data_dir='/custom/path',
            img_size=(128, 128),
            batch_size=16
        )
        
        self.assertEqual(loader.data_dir, '/custom/path')
        self.assertEqual(loader.img_size, (128, 128))
        self.assertEqual(loader.batch_size, 16)
    
    def create_dummy_dataset(self):
        """
        Create dummy dataset structure for testing
        """
        # Create train directory structure
        train_dir = os.path.join(self.temp_dir, 'train')
        os.makedirs(train_dir)
        
        # Create some sport classes
        sports = ['tennis', 'football', 'basketball']
        for sport in sports:
            sport_dir = os.path.join(train_dir, sport)
            os.makedirs(sport_dir)
            
            # Create dummy image files
            for i in range(5):
                img_path = os.path.join(sport_dir, f'img_{i}.jpg')
                with open(img_path, 'w') as f:
                    f.write('dummy image data')
        
        # Create test directory structure
        test_dir = os.path.join(self.temp_dir, 'test')
        os.makedirs(test_dir)
        
        for sport in sports:
            sport_dir = os.path.join(test_dir, sport)
            os.makedirs(sport_dir)
            
            # Create dummy test images
            for i in range(2):
                img_path = os.path.join(sport_dir, f'test_{i}.jpg')
                with open(img_path, 'w') as f:
                    f.write('dummy test image data')
    
    def test_load_dataset_info(self):
        """
        Test loading dataset information
        """
        self.create_dummy_dataset()
        
        train_df, val_df, test_df = self.data_loader.load_dataset_info()
        
        # Check that DataFrames are created
        self.assertIsNotNone(train_df)
        self.assertIsNotNone(val_df)
        self.assertIsNotNone(test_df)
        
        # Check DataFrame contents
        self.assertTrue(len(train_df) > 0)
        self.assertTrue(len(val_df) > 0)
        self.assertTrue(len(test_df) > 0)
        
        # Check columns
        expected_columns = ['filepath', 'label']
        self.assertListEqual(list(train_df.columns), expected_columns)
        self.assertListEqual(list(val_df.columns), expected_columns)
        self.assertListEqual(list(test_df.columns), expected_columns)
    
    def test_load_dataset_info_no_data(self):
        """
        Test loading dataset info when no data exists
        """
        # This should handle gracefully when no train directory exists
        train_df, val_df, test_df = self.data_loader.load_dataset_info()
        
        # Should create empty DataFrames or handle appropriately
        self.assertIsNotNone(train_df)
        self.assertIsNotNone(val_df)
    
    @patch('matplotlib.pyplot.show')
    def test_analyze_class_distribution(self, mock_show):
        """
        Test class distribution analysis
        """
        self.create_dummy_dataset()
        self.data_loader.load_dataset_info()
        
        # This should not raise an exception
        self.data_loader.analyze_class_distribution()
        
        # Check that matplotlib show was called
        mock_show.assert_called()
    
    def test_analyze_class_distribution_no_data(self):
        """
        Test class distribution analysis with no data loaded
        """
        # Should handle gracefully when no data is loaded
        with patch('builtins.print') as mock_print:
            self.data_loader.analyze_class_distribution()
            mock_print.assert_called_with("Please load dataset info first using load_dataset_info()")


class TestDataLoaderFunctions(unittest.TestCase):
    """
    Test cases for data loader utility functions
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """
        Clean up test fixtures
        """
        shutil.rmtree(self.temp_dir)
    
    @patch('src.data_loader.SportsDataLoader')
    def test_load_sports_data(self, mock_loader_class):
        """
        Test load_sports_data convenience function
        """
        from src.data_loader import load_sports_data
        
        # Mock the SportsDataLoader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Mock the return values
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_test_gen = MagicMock()
        mock_loader.create_data_generators.return_value = (mock_train_gen, mock_val_gen, mock_test_gen)
        
        # Call the function
        result = load_sports_data(
            data_dir=self.temp_dir,
            img_size=(224, 224),
            batch_size=32,
            augmentation=True
        )
        
        # Check that SportsDataLoader was created with correct parameters
        mock_loader_class.assert_called_once_with(self.temp_dir, (224, 224), 32)
        
        # Check that load_dataset_info was called
        mock_loader.load_dataset_info.assert_called_once()
        
        # Check that create_data_generators was called
        mock_loader.create_data_generators.assert_called_once_with(True)
        
        # Check return value
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], mock_loader)
        self.assertEqual(result[1], mock_train_gen)
        self.assertEqual(result[2], mock_val_gen)
        self.assertEqual(result[3], mock_test_gen)


if __name__ == '__main__':
    unittest.main()