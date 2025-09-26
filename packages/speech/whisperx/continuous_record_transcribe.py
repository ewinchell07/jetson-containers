import sounddevice as sd
import numpy as np
import wave
import threading
import time
import os
from datetime import datetime
import whisperx
import queue
import torch
import json
import pandas as pd
from pyannote.audio import Pipeline

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, dtype=np.int16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_queue = queue.Queue()
        self.current_chunk = []
        self.overflow_count = 0
        self.last_overflow_time = time.time()
        
        # Load WhisperX model
        print("Loading WhisperX model...")
        self.model = whisperx.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float32")
        print(torch.cuda.is_available())

        # Load diarization pipeline
        print("Loading diarization pipeline...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Create recordings directory if it doesn't exist
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # List available audio devices
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        
        # Use Samson Go Mic as input device
        self.input_device = 0  # Samson Go Mic: USB Audio (hw:0,0)
        print(f"\nUsing input device: {sd.query_devices(self.input_device)['name']}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            current_time = time.time()
            if status.input_overflow:
                self.overflow_count += 1
                # Only print warning if it's been at least 5 seconds since last overflow
                if current_time - self.last_overflow_time > 5:
                    print(f"Warning: Input buffer overflow detected. This may cause audio loss. (Overflow count: {self.overflow_count})")
                    self.last_overflow_time = current_time
            else:
                print(f"Status: {status}")
                
        if self.recording:
            # Check queue size and print warning if it's getting too large
            if self.audio_queue.qsize() > 100:  # Arbitrary threshold
                print(f"Warning: Audio queue is getting large ({self.audio_queue.qsize()} chunks). Processing may be falling behind.")
            
            self.current_chunk.extend(indata.copy())
            self.audio_queue.put(indata.copy())

    def save_audio_chunk(self, audio_data, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def transcribe_audio(self, audio_file):
        try:
            # Transcribe with WhisperX
            result = self.model.transcribe(audio_file)
            
            # Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda" if torch.cuda.is_available() else "cpu")
            result = whisperx.align(result["segments"], model_a, metadata, audio_file, "cuda" if torch.cuda.is_available() else "cpu")
            
            # Perform speaker diarization
            diarize_segments = self.diarization_pipeline(audio_file)
            
            # Convert diarization segments to DataFrame format expected by whisperx
            diarize_segments_list = []
            for segment, track, speaker in diarize_segments.itertracks(yield_label=True):
                diarize_segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker
                })
            
            diarize_df = pd.DataFrame(diarize_segments_list)
            
            # Assign speaker labels
            result = whisperx.assign_word_speakers(diarize_df, result)
            
            return result
        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
            return None

    def process_audio_chunks(self):
        while self.recording:
            # Collect 10 minutes of audio
            chunk_duration = 600  # 10 minutes in seconds
            samples_per_chunk = chunk_duration * self.sample_rate
            audio_chunk = []
            
            print(f"\nCollecting {chunk_duration} seconds of audio...")
            while len(audio_chunk) < samples_per_chunk and self.recording:
                try:
                    data = self.audio_queue.get(timeout=1)
                    audio_chunk.extend(data)
                except queue.Empty:
                    print("No audio data received in the last second")
                    continue

            if audio_chunk:
                # Convert to numpy array
                audio_data = np.array(audio_chunk, dtype=self.dtype)
                
                # Check if we have any non-zero audio
                if np.max(np.abs(audio_data)) < 0.01:
                    print("\nNo audio detected in this chunk, skipping...")
                    continue
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = os.path.join(self.recordings_dir, f"recording_{timestamp}.wav")
                
                # Save the audio file
                self.save_audio_chunk(audio_data, audio_file)
                print(f"\nSaved audio chunk to {audio_file}")
                
                # Transcribe the audio
                print("Transcribing audio...")
                result = self.transcribe_audio(audio_file)
                if result:
                    # Save transcription with speaker labels
                    transcript_file = os.path.join(self.recordings_dir, f"transcript_{timestamp}.json")
                    with open(transcript_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Transcription saved to {transcript_file}")
                    
                    # Print transcription with speaker labels
                    print("\nTranscription with speaker labels:")
                    for segment in result["segments"]:
                        speaker = segment.get("speaker", "UNKNOWN")
                        text = segment["text"]
                        print(f"Speaker {speaker}: {text}")

    def start_recording(self):
        self.recording = True
        self.current_chunk = []
        self.overflow_count = 0
        self.last_overflow_time = time.time()
        
        # Start the audio stream with larger blocksize and higher latency
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self.audio_callback,
                blocksize=2048,  # Increased from 1024
                latency='high',   # Added high latency setting
                device=self.input_device
            )
            self.stream.start()
            print("\nAudio stream started successfully")
        except Exception as e:
            print(f"Error starting audio stream: {str(e)}")
            return
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self.process_audio_chunks)
        self.process_thread.start()
        
        print("Recording started... Press Ctrl+C to stop.")

    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        print("Recording stopped.")

def main():
    recorder = AudioRecorder()
    try:
        recorder.start_recording()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
        recorder.stop_recording()

if __name__ == "__main__":
    main() 