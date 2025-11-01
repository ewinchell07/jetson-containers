#!/usr/bin/env python3
"""
Test script to verify the 500Hz gain noise filtering functionality.
This creates a test audio signal with 500Hz and its harmonics, then applies the filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os

# Add the current directory to path to import the recorder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_recorder import AudioConfig, ContinuousRecorder

def create_test_signal(duration=2.0, sample_rate=48000):
    """Create a test signal with 500Hz gain noise and harmonics"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create signal with 500Hz and its harmonics up to 6000Hz
    test_signal = np.zeros_like(t)
    base_freq = 500.0
    max_freq = 6000.0
    
    freq = base_freq
    while freq <= max_freq:
        # Add sine wave with decreasing amplitude for higher harmonics
        amplitude = 1.0 / (freq / base_freq)  # Decreasing amplitude
        test_signal += amplitude * np.sin(2 * np.pi * freq * t)
        freq += base_freq
    
    # Add some background noise
    test_signal += 0.1 * np.random.randn(len(t))
    
    return t, test_signal

def test_gain_noise_filtering():
    """Test the gain noise filtering functionality"""
    print("🧪 Testing 500Hz gain noise filtering...")
    
    # Create test configuration
    config = AudioConfig(
        native_sample_rate=48000,
        target_sample_rate=16000,
        channels=1,
        enable_gain_noise_filtering=True,
        gain_noise_base_freq=500.0,
        gain_noise_max_freq=6000.0,
        gain_noise_q=30.0,
        enable_noise_filtering=False,  # Disable other filtering for this test
        enable_amplification=False     # Disable amplification for this test
    )
    
    # Create recorder instance
    recorder = ContinuousRecorder(config, "test_output")
    
    # Generate test signal
    print("📊 Generating test signal with 500Hz whine and harmonics...")
    t, original_signal = create_test_signal()
    
    # Apply gain noise filtering
    print("🔇 Applying gain noise filtering...")
    filtered_signal = recorder._filter_audio(original_signal, 48000)
    
    # Calculate frequency spectra
    print("📈 Analyzing frequency content...")
    
    # Original signal FFT
    fft_original = np.fft.fft(original_signal)
    freqs = np.fft.fftfreq(len(original_signal), 1/48000)
    
    # Filtered signal FFT
    fft_filtered = np.fft.fft(filtered_signal)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Time domain
    ax1.plot(t[:int(0.1*48000)], original_signal[:int(0.1*48000)], 'b-', label='Original', alpha=0.7)
    ax1.plot(t[:int(0.1*48000)], filtered_signal[:int(0.1*48000)], 'r-', label='Filtered', alpha=0.7)
    ax1.set_title('Time Domain (first 0.1 seconds)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)
    
    # Frequency domain - original
    positive_freqs = freqs[:len(freqs)//2]
    ax2.semilogy(positive_freqs, np.abs(fft_original[:len(fft_original)//2]), 'b-', label='Original')
    ax2.set_title('Frequency Spectrum - Original')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 8000)
    ax2.legend()
    ax2.grid(True)
    
    # Frequency domain - filtered
    ax3.semilogy(positive_freqs, np.abs(fft_filtered[:len(fft_filtered)//2]), 'r-', label='Filtered')
    ax3.set_title('Frequency Spectrum - Filtered')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(0, 8000)
    ax3.legend()
    ax3.grid(True)
    
    # Highlight the frequencies that should be filtered
    for ax in [ax2, ax3]:
        for freq in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]:
            if freq <= 6000:
                ax.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig('whine_filter_test.png', dpi=150, bbox_inches='tight')
    print("📊 Results saved to 'whine_filter_test.png'")
    
    # Calculate reduction at target frequencies
    print("\n📊 Gain noise reduction analysis:")
    target_freqs = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    
    for target_freq in target_freqs:
        if target_freq <= 6000:
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(positive_freqs - target_freq))
            
            original_mag = np.abs(fft_original[freq_idx])
            filtered_mag = np.abs(fft_filtered[freq_idx])
            
            if original_mag > 0:
                reduction_db = 20 * np.log10(filtered_mag / original_mag)
                print(f"  {target_freq:4d}Hz: {reduction_db:6.1f}dB reduction")
            else:
                print(f"  {target_freq:4d}Hz: No signal detected")
    
    print("\n✅ Gain noise filtering test completed!")
    print("📁 Check 'whine_filter_test.png' for visual results")

if __name__ == "__main__":
    test_gain_noise_filtering()
