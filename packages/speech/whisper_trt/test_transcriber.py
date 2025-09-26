#!/usr/bin/env python3
"""
Comprehensive unit tests for the whisper_trt transcriber module.
Tests all major components including GPU management, model loading, transcription, and diarization.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcriber import (
    GPUManager, ModelManager, DiarizationManager, Transcriber,
    get_transcription_config, get_diarization_config_local, get_audio_config_local,
    DIARIZATION_AVAILABLE
)


class TestGPUManager:
    """Test GPU memory management functionality"""
    
    def test_cleanup_memory(self):
        """Test GPU memory cleanup"""
        # This should not raise an exception
        GPUManager.cleanup_memory()
        assert True  # If we get here, cleanup worked
    
    def test_get_memory_usage(self):
        """Test GPU memory usage reporting"""
        usage = GPUManager.get_memory_usage()
        assert isinstance(usage, (int, float))
        assert usage >= 0
    
    @patch('torch.cuda.is_available')
    def test_cleanup_memory_no_cuda(self, mock_cuda_available):
        """Test cleanup when CUDA is not available"""
        mock_cuda_available.return_value = False
        GPUManager.cleanup_memory()  # Should not raise exception
        assert True


class TestModelManager:
    """Test model loading and management"""
    
    @patch('whisper.load_model')
    @patch('torch.cuda.is_available')
    def test_model_loading_regular_whisper(self, mock_cuda_available, mock_load_model):
        """Test loading regular Whisper model"""
        mock_cuda_available.return_value = True
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        manager = ModelManager("tiny.en")
        
        assert manager.model_name == "tiny.en"
        assert manager.model == mock_model
        mock_load_model.assert_called_once_with("tiny.en")
    
    @patch('whisper.load_model')
    @patch('whisper_trt.load_trt_model')
    @patch('torch.cuda.is_available')
    def test_model_loading_fallback_to_trt(self, mock_cuda_available, mock_load_trt, mock_load_whisper):
        """Test fallback to Whisper-TRT when regular Whisper fails"""
        mock_cuda_available.return_value = True
        mock_load_whisper.side_effect = Exception("Regular Whisper failed")
        mock_trt_model = Mock()
        mock_load_trt.return_value = mock_trt_model
        
        manager = ModelManager("tiny.en")
        
        assert manager.model == mock_trt_model
        mock_load_trt.assert_called_once_with("tiny.en", verbose=True)
    
    @patch('whisper.load_model')
    def test_model_loading_failure(self, mock_load_model):
        """Test handling of model loading failure"""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception):
            ModelManager("tiny.en")
    
    @patch('whisper.load_model')
    @patch('torch.cuda.is_available')
    def test_transcribe_audio_success(self, mock_cuda_available, mock_load_model):
        """Test successful audio transcription"""
        mock_cuda_available.return_value = True
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Hello world',
            'segments': [{'start': 0, 'end': 1, 'text': 'Hello world'}]
        }
        mock_load_model.return_value = mock_model
        
        manager = ModelManager("tiny.en")
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            # Create a simple audio file (1 second of silence)
            sample_rate = 16000
            duration = 1
            audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            try:
                result = manager.transcribe_audio(tmp_file.name)
                assert result['text'] == 'Hello world'
                assert len(result['segments']) == 1
            finally:
                os.unlink(tmp_file.name)
    
    @patch('whisper.load_model')
    def test_transcribe_audio_empty_result(self, mock_load_model):
        """Test handling of empty transcription result"""
        mock_model = Mock()
        mock_model.transcribe.return_value = {'text': '', 'segments': []}
        mock_load_model.return_value = mock_model
        
        manager = ModelManager("tiny.en")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sample_rate = 16000
            duration = 1
            audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            try:
                result = manager.transcribe_audio(tmp_file.name)
                assert result['text'] == ''
                assert result['segments'] == []
            finally:
                os.unlink(tmp_file.name)


class TestDiarizationManager:
    """Test speaker diarization functionality"""
    
    @pytest.mark.skipif(not DIARIZATION_AVAILABLE, reason="Resemblyzer not available")
    def test_diarization_manager_initialization(self):
        """Test DiarizationManager initialization"""
        manager = DiarizationManager()
        assert manager.encoder is not None
    
    @pytest.mark.skipif(not DIARIZATION_AVAILABLE, reason="Resemblyzer not available")
    def test_segment_audio(self):
        """Test audio segmentation"""
        manager = DiarizationManager()
        
        # Create test audio data
        sample_rate = 16000
        duration = 10  # 10 seconds
        wav = np.random.randn(sample_rate * duration).astype(np.float32)
        
        segments = manager._segment_audio(wav, sample_rate)
        
        assert len(segments) > 0
        for segment in segments:
            assert 'start' in segment
            assert 'end' in segment
            assert 'wav' in segment
            assert segment['start'] < segment['end']
    
    @pytest.mark.skipif(not DIARIZATION_AVAILABLE, reason="Resemblyzer not available")
    def test_cluster_speakers(self):
        """Test speaker clustering"""
        manager = DiarizationManager()
        
        # Create test embeddings (simulate 5 segments, 2 speakers)
        embeddings = np.random.randn(5, 256).astype(np.float32)
        
        speaker_labels = manager._cluster_speakers(embeddings, num_speakers=2)
        
        assert len(speaker_labels) == 5
        assert len(set(speaker_labels)) <= 2  # Should have at most 2 speakers
    
    @pytest.mark.skipif(not DIARIZATION_AVAILABLE, reason="Resemblyzer not available")
    def test_process_audio_with_mock_file(self):
        """Test audio processing with a mock audio file"""
        manager = DiarizationManager()
        
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sample_rate = 16000
            duration = 5  # 5 seconds
            audio_data = np.random.randn(sample_rate * duration).astype(np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            try:
                speaker_segments = manager.process_audio(tmp_file.name, num_speakers=2)
                
                assert isinstance(speaker_segments, list)
                for segment in speaker_segments:
                    assert 'start' in segment
                    assert 'end' in segment
                    assert 'speaker' in segment
                    assert segment['speaker'].startswith('SPEAKER_')
            finally:
                os.unlink(tmp_file.name)


class TestTranscriber:
    """Test main Transcriber class"""
    
    @patch('transcriber.ModelManager')
    def test_transcriber_initialization_with_diarization(self, mock_model_manager):
        """Test Transcriber initialization with diarization enabled"""
        mock_model_manager.return_value = Mock()
        
        transcriber = Transcriber("tiny.en", enable_diarization=True)
        
        assert transcriber.model_manager is not None
        # Diarization manager may be None if Resemblyzer is not available
        assert transcriber.diarization_manager is None or hasattr(transcriber.diarization_manager, 'encoder')
    
    @patch('transcriber.ModelManager')
    def test_transcriber_initialization_without_diarization(self, mock_model_manager):
        """Test Transcriber initialization with diarization disabled"""
        mock_model_manager.return_value = Mock()
        
        transcriber = Transcriber("tiny.en", enable_diarization=False)
        
        assert transcriber.model_manager is not None
        assert transcriber.diarization_manager is None
    
    @patch('transcriber.ModelManager')
    @patch('transcriber.DiarizationManager')
    def test_merge_transcription_with_speakers(self, mock_diarization_manager, mock_model_manager):
        """Test merging transcription with speaker information"""
        mock_model_manager.return_value = Mock()
        mock_diarization_manager.return_value = Mock()
        
        transcriber = Transcriber("tiny.en", enable_diarization=False)
        
        # Mock transcription result
        transcription = {
            'text': 'Hello world',
            'segments': [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello'},
                {'start': 1.0, 'end': 2.0, 'text': 'world'}
            ]
        }
        
        # Mock speaker segments
        speaker_segments = [
            {'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'},
            {'start': 1.5, 'end': 2.0, 'speaker': 'SPEAKER_01'}
        ]
        
        merged = transcriber._merge_transcription_with_speakers(transcription, speaker_segments)
        
        assert len(merged) == 2
        assert merged[0]['speaker'] == 'SPEAKER_00'
        assert merged[1]['speaker'] == 'SPEAKER_01'
        assert 'speaker_confidence' in merged[0]
        assert 'speaker_confidence' in merged[1]
    
    @patch('transcriber.ModelManager')
    def test_calculate_speaker_confidence(self, mock_model_manager):
        """Test speaker confidence calculation"""
        mock_model_manager.return_value = Mock()
        
        transcriber = Transcriber("tiny.en", enable_diarization=False)
        
        speaker_segments = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 1.0, 'end': 2.0, 'speaker': 'SPEAKER_01'}
        ]
        
        # Test perfect overlap
        confidence = transcriber._calculate_speaker_confidence(0.0, 1.0, speaker_segments, 'SPEAKER_00')
        assert confidence == 1.0
        
        # Test partial overlap
        confidence = transcriber._calculate_speaker_confidence(0.5, 1.5, speaker_segments, 'SPEAKER_00')
        assert 0.0 < confidence < 1.0
        
        # Test no overlap
        confidence = transcriber._calculate_speaker_confidence(0.0, 1.0, speaker_segments, 'SPEAKER_01')
        assert confidence == 0.0
    
    @patch('transcriber.ModelManager')
    def test_deduplicate_speaker_segments(self, mock_model_manager):
        """Test speaker segment deduplication"""
        mock_model_manager.return_value = Mock()
        
        transcriber = Transcriber("tiny.en", enable_diarization=False)
        
        # Test overlapping segments with same speaker
        speaker_segments = [
            {'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'},
            {'start': 0.5, 'end': 1.5, 'speaker': 'SPEAKER_00'},
            {'start': 2.0, 'end': 3.0, 'speaker': 'SPEAKER_01'}
        ]
        
        deduplicated = transcriber._deduplicate_speaker_segments(speaker_segments)
        
        assert len(deduplicated) == 2  # Should merge first two segments
        assert deduplicated[0]['start'] == 0.0
        assert deduplicated[0]['end'] == 1.5  # Merged end time
        assert deduplicated[1]['speaker'] == 'SPEAKER_01'
    
    @patch('transcriber.ModelManager')
    @patch('transcriber.DiarizationManager')
    def test_process_file_success(self, mock_diarization_manager, mock_model_manager):
        """Test successful file processing"""
        # Mock model manager
        mock_model = Mock()
        mock_model.transcribe_audio.return_value = {
            'text': 'Test transcription',
            'segments': [{'start': 0, 'end': 1, 'text': 'Test transcription'}]
        }
        mock_model_manager.return_value = mock_model
        
        # Mock diarization manager
        mock_diarization_manager.return_value = Mock()
        
        transcriber = Transcriber("tiny.en", enable_diarization=True)
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sample_rate = 16000
            duration = 1
            audio_data = np.zeros(sample_rate * duration, dtype=np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                try:
                    result = transcriber.process_file(tmp_file.name, output_dir)
                    
                    assert 'transcription' in result
                    assert 'timestamp' in result
                    assert 'audio_file' in result
                    assert 'model' in result
                    assert result['transcription']['text'] == 'Test transcription'
                finally:
                    os.unlink(tmp_file.name)


class TestConfiguration:
    """Test configuration functions"""
    
    def test_get_transcription_config(self):
        """Test transcription configuration"""
        config = get_transcription_config()
        
        assert isinstance(config, dict)
        assert 'temperature' in config
        assert 'beam_size' in config
        assert 'language' in config
        assert 'task' in config
        assert config['task'] == 'transcribe'
    
    def test_get_diarization_config_local(self):
        """Test diarization configuration"""
        config = get_diarization_config_local()
        
        assert isinstance(config, dict)
        assert 'segment_duration' in config
        assert 'overlap_duration' in config
        assert 'min_speakers' in config
        assert 'max_speakers' in config
    
    def test_get_audio_config_local(self):
        """Test audio configuration"""
        config = get_audio_config_local()
        
        assert isinstance(config, dict)
        assert 'target_sample_rate' in config
        assert isinstance(config['target_sample_rate'], int)


class TestIntegration:
    """Integration tests that test multiple components together"""
    
    @patch('transcriber.ModelManager')
    @patch('transcriber.DiarizationManager')
    def test_end_to_end_processing(self, mock_diarization_manager, mock_model_manager):
        """Test end-to-end processing workflow"""
        # Mock model manager
        mock_model = Mock()
        mock_model.transcribe_audio.return_value = {
            'text': 'Hello world test',
            'segments': [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello'},
                {'start': 1.0, 'end': 2.0, 'text': 'world'},
                {'start': 2.0, 'end': 3.0, 'text': 'test'}
            ]
        }
        mock_model_manager.return_value = mock_model
        
        # Mock diarization manager
        mock_diarization = Mock()
        mock_diarization.process_audio.return_value = [
            {'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'},
            {'start': 1.5, 'end': 3.0, 'speaker': 'SPEAKER_01'}
        ]
        mock_diarization_manager.return_value = mock_diarization
        
        transcriber = Transcriber("tiny.en", enable_diarization=True)
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sample_rate = 16000
            duration = 3
            audio_data = np.random.randn(sample_rate * duration).astype(np.float32)
            import soundfile as sf
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                try:
                    result = transcriber.process_file(tmp_file.name, output_dir)
                    
                    # Verify the result structure
                    assert 'transcription' in result
                    assert 'speaker_segments' in result
                    assert 'merged_segments' in result
                    assert 'gpu_memory_used_gb' in result
                    assert 'config' in result
                    
                    # Verify transcription
                    assert result['transcription']['text'] == 'Hello world test'
                    assert len(result['transcription']['segments']) == 3
                    
                    # Verify speaker segments
                    assert len(result['speaker_segments']) == 2
                    
                    # Verify merged segments have speaker information
                    assert len(result['merged_segments']) == 3
                    for segment in result['merged_segments']:
                        assert 'speaker' in segment
                        assert 'speaker_confidence' in segment
                    
                    # Verify output file was created
                    output_files = list(Path(output_dir).glob('*.json'))
                    assert len(output_files) == 1
                    
                    # Verify JSON content
                    with open(output_files[0], 'r') as f:
                        saved_data = json.load(f)
                    assert saved_data['transcription']['text'] == 'Hello world test'
                    
                finally:
                    os.unlink(tmp_file.name)


def run_tests():
    """Run all tests and return results"""
    print("Running whisper_trt transcriber tests...")
    
    # Run pytest with verbose output
    result = pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--disable-warnings'
    ])
    
    return result


if __name__ == "__main__":
    exit(run_tests())

