#!/usr/bin/env python3
"""
Audio Analysis Demo

Demonstrates the new audio quality analysis features in transcriber.py
that can detect if audio files contain speech before attempting transcription.
"""

import sys
import logging
from pathlib import Path
from transcriber import AudioProcessor

def analyze_audio_file(audio_file: str):
    """Analyze a single audio file and show results"""
    print(f"\n{'='*60}")
    print(f"🔍 ANALYZING: {audio_file}")
    print(f"{'='*60}")
    
    try:
        processor = AudioProcessor()
        should_transcribe, analysis = processor.should_transcribe(audio_file)
        
        print(f"📊 ANALYSIS RESULTS:")
        print(f"   Duration: {analysis.get('duration_seconds', 0):.2f} seconds")
        print(f"   RMS Energy: {analysis.get('rms_energy', 0):.4f}")
        print(f"   Max Amplitude: {analysis.get('max_amplitude', 0):.4f}")
        print(f"   Dynamic Range: {analysis.get('dynamic_range', 0):.4f}")
        print(f"   Speech Ratio: {analysis.get('speech_ratio', 0):.3f}")
        print(f"   Confidence Score: {analysis.get('confidence_score', 0):.3f}")
        print(f"   Recommendation: {analysis.get('recommendation', 'Unknown')}")
        print(f"   Should Transcribe: {'✅ YES' if should_transcribe else '❌ NO'}")
        
        # Show detailed characteristics
        if 'avg_spectral_centroid' in analysis:
            print(f"\n📈 SPECTRAL ANALYSIS:")
            print(f"   Spectral Centroid: {analysis.get('avg_spectral_centroid', 0):.1f} Hz")
            print(f"   Spectral Rolloff: {analysis.get('avg_spectral_rolloff', 0):.1f} Hz")
            print(f"   Zero Crossing Rate: {analysis.get('avg_zero_crossing_rate', 0):.4f}")
        
        # Show flags
        print(f"\n🚩 QUALITY FLAGS:")
        print(f"   Is Quiet: {'⚠️  YES' if analysis.get('is_quiet', False) else '✅ NO'}")
        print(f"   Is Too Short: {'⚠️  YES' if analysis.get('is_too_short', False) else '✅ NO'}")
        print(f"   Has No Dynamics: {'⚠️  YES' if analysis.get('has_no_dynamics', False) else '✅ NO'}")
        print(f"   Has Speech Characteristics: {'✅ YES' if analysis.get('has_speech_characteristics', False) else '❌ NO'}")
        print(f"   Is Likely Speech: {'✅ YES' if analysis.get('is_likely_speech', False) else '❌ NO'}")
        
        return should_transcribe, analysis
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, {"error": str(e)}

def main():
    """Main demo function"""
    print("🎤 AUDIO ANALYSIS DEMO")
    print("=" * 60)
    print("This demo shows how the transcriber can analyze audio files")
    print("to detect if they contain speech before attempting transcription.")
    print()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if len(sys.argv) < 2:
        print("Usage: python3 audio_analysis_demo.py <audio_file> [audio_file2] ...")
        print("\nExample:")
        print("  python3 audio_analysis_demo.py recordings/test.wav")
        print("  python3 audio_analysis_demo.py recordings/*.wav")
        return 1
    
    audio_files = sys.argv[1:]
    results = []
    
    for audio_file in audio_files:
        if not Path(audio_file).exists():
            print(f"❌ File not found: {audio_file}")
            continue
            
        should_transcribe, analysis = analyze_audio_file(audio_file)
        results.append((audio_file, should_transcribe, analysis))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    
    total_files = len(results)
    files_to_transcribe = sum(1 for _, should_transcribe, _ in results if should_transcribe)
    files_to_skip = total_files - files_to_transcribe
    
    print(f"Total files analyzed: {total_files}")
    print(f"Files to transcribe: {files_to_transcribe}")
    print(f"Files to skip: {files_to_skip}")
    
    if files_to_skip > 0:
        print(f"\n⏭️  FILES TO SKIP:")
        for audio_file, should_transcribe, analysis in results:
            if not should_transcribe:
                reason = analysis.get('recommendation', 'Unknown reason')
                print(f"   {Path(audio_file).name}: {reason}")
    
    print(f"\n💡 TIP: Use --skip-analysis flag with transcriber.py to bypass this analysis")
    print(f"💡 TIP: Adjust --min-confidence and --min-duration thresholds as needed")
    
    return 0

if __name__ == "__main__":
    exit(main())
