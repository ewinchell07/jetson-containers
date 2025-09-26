import sounddevice as sd
import numpy as np
import wave
import threading
import time
import os
from datetime import datetime
import whisper
import queue
import torch

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, dtype=np.int16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_queue = queue.Queue()
        self.current_chunk = []
        self.model = whisper.load_model("base")
        
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
            print(f"Status: {status}")
        if self.recording:
            # Print audio levels for debugging
            if np.max(np.abs(indata)) > 0.1:  # Only print if there's significant audio
                print(f"Audio level: {np.max(np.abs(indata)):.3f}")
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
            result = self.model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing {audio_file}: {str(e)}")
            return None

    def process_audio_chunks(self):
        while self.recording:
            # Collect 5 seconds of audio
            chunk_duration = 5  # 5 seconds
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
                transcription = self.transcribe_audio(audio_file)
                if transcription:
                    # Save transcription
                    transcript_file = os.path.join(self.recordings_dir, f"transcript_{timestamp}.txt")
                    with open(transcript_file, 'w') as f:
                        f.write(transcription)
                    print(f"Transcription saved to {transcript_file}")
                    print(f"Transcription: {transcription}")

    def start_recording(self):
        self.recording = True
        self.current_chunk = []
        
        # Start the audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self.audio_callback,
                blocksize=1024,  # Smaller block size for more frequent callbacks
                device=self.input_device  # Use Samson Go Mic
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