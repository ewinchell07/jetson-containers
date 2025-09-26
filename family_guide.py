#!/usr/bin/env python3
"""
FamilyGuide - Simple nano_llm-based family coaching assistant
Usage: python3 family_guide.py <transcript_folder> "<your question>"
"""

import json
import os
import sys
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse

# System prompt for FamilyGuide
FAMILY_GUIDE_PROMPT = """You are FamilyGuide, a small on-device assistant that helps parents raise capable, happy children.

Principles: parents first; build knowledge → agency; avoid cognitive offloading; encourage strong family routines; ground answers in provided transcripts.

RAG use: You may receive transcript snippets with timestamps and speaker. Prefer the most recent, most relevant snippets. If unsure, say so and suggest next steps. Cite any claim with its time range like [14:26–14:42].

Safety & limits: Respect off-limits topics: [medical advice, legal advice, therapy recommendations]. If asked, gently refuse and offer safer alternatives. Do not guess identities. No web calls.

Style: warm, concise, non-judgmental; plain English.

Format: 2–3 sentence answer → Why this helps (one line) → Try this (1–3 concrete steps, 10–20 words each).

Coaching patterns: label emotion + offer a choice; scaffold language; propose small agentic tasks; tie advice to daily routines.

Contract: keep under 200 words; always cite transcripts when used; end with "Try this:"."""

def load_transcript(file_path: str) -> Dict[str, Any]:
    """Load a transcript JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_recent_segments(transcript: Dict[str, Any], hours_back: int = 24) -> List[Dict[str, Any]]:
    """Extract segments from the last N hours of a transcript."""
    if not transcript or 'transcription' not in transcript:
        return []
    
    segments = transcript['transcription'].get('segments', [])
    if not segments:
        return []
    
    # Get the latest timestamp from the transcript
    transcript_time = datetime.fromisoformat(transcript['timestamp'].replace('Z', '+00:00'))
    cutoff_time = transcript_time - timedelta(hours=hours_back)
    
    recent_segments = []
    for segment in segments:
        # Convert segment start time to datetime
        segment_start = transcript_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(seconds=segment['start'])
        
        if segment_start >= cutoff_time:
            recent_segments.append(segment)
    
    return recent_segments

def format_segment_for_rag(segment: Dict[str, Any], transcript_file: str) -> str:
    """Format a transcript segment for RAG input."""
    start_time = segment['start']
    end_time = segment['end']
    text = segment['text'].strip()
    speaker = segment.get('speaker', 'UNKNOWN')
    
    # Format time as MM:SS
    start_mm_ss = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
    end_mm_ss = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
    
    return f"[{start_mm_ss}–{end_mm_ss}] {speaker}: {text}"

def get_recent_transcripts(transcript_folder: str, days_back: int = 7) -> List[str]:
    """Get transcript files from the last N days."""
    transcript_files = []
    
    # Get all JSON files in the folder
    pattern = os.path.join(transcript_folder, "*.json")
    all_files = glob.glob(pattern)
    
    # Filter by date (files with date in filename)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        # Extract date from filename like "transcript_recording_20250914_010604_20250914_121936.json"
        try:
            date_part = filename.split('_')[2]  # Get the date part
            file_date = datetime.strptime(date_part, '%Y%m%d')
            if file_date >= cutoff_date:
                transcript_files.append(file_path)
        except (IndexError, ValueError):
            # If we can't parse the date, include the file anyway
            transcript_files.append(file_path)
    
    # Sort by modification time (newest first)
    transcript_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return transcript_files

def build_rag_context(transcript_folder: str, max_segments: int = 20) -> str:
    """Build RAG context from recent transcripts."""
    recent_files = get_recent_transcripts(transcript_folder, days_back=7)
    
    all_segments = []
    for file_path in recent_files[:5]:  # Limit to 5 most recent files
        transcript = load_transcript(file_path)
        if transcript:
            segments = extract_recent_segments(transcript, hours_back=24)
            for segment in segments:
                formatted_segment = format_segment_for_rag(segment, os.path.basename(file_path))
                all_segments.append(formatted_segment)
    
    # Take the most recent segments
    recent_segments = all_segments[:max_segments]
    
    if not recent_segments:
        return "No recent transcript data available."
    
    context = "Recent family conversations:\n" + "\n".join(recent_segments)
    return context

def analyze_transcripts_basic(question: str, rag_context: str) -> str:
    """Basic transcript analysis when nano_llm is not available."""
    if "No recent transcript data available" in rag_context:
        return f"""I don't see any recent transcript data to analyze.

Why this helps: Without conversation data, I can't provide specific insights.

Try this: 
1. Check that transcript files are in the correct folder
2. Ensure recordings are recent (within the last 7 days)
3. Verify the transcript files contain conversation data

[Note: This is a basic analysis - nano_llm not available]"""
    
    # Count speakers and basic stats
    lines = rag_context.split('\n')[1:]  # Skip the header
    speaker_counts = {}
    total_segments = 0
    
    for line in lines:
        if ']:' in line and line.strip():
            total_segments += 1
            # Extract speaker
            if 'SPEAKER_' in line:
                speaker = line.split('SPEAKER_')[1].split(':')[0]
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    # Generate basic insights
    insights = []
    if total_segments > 0:
        insights.append(f"I found {total_segments} conversation segments")
    
    if speaker_counts:
        most_active = max(speaker_counts, key=speaker_counts.get)
        insights.append(f"SPEAKER_{most_active} was most active with {speaker_counts[most_active]} segments")
    
    # Look for common patterns
    if "okay" in rag_context.lower() or "ok" in rag_context.lower():
        insights.append("I notice frequent use of 'okay' - this suggests collaborative communication")
    
    if "?" in rag_context:
        insights.append("I see questions being asked - this indicates interactive dialogue")
    
    if "no" in rag_context.lower() or "yes" in rag_context.lower():
        insights.append("I notice direct responses - this shows clear communication")
    
    response = f"""Based on the recent conversations, here's my guidance:

{question}

Analysis: {'; '.join(insights) if insights else 'Basic conversation patterns detected'}

Why this helps: Understanding communication patterns helps build stronger family connections.

Try this: 
1. Notice when family members express emotions
2. Ask open-ended questions about their day  
3. Create a daily check-in routine

[Note: This is basic analysis - nano_llm not available. For deeper insights, the nano_llm container needs to be rebuilt.]"""
    
    return response

def run_nano_llm_chat(question: str, rag_context: str) -> str:
    """Run nano_llm chat with the question and RAG context."""
    # Build the full prompt
    full_prompt = f"{FAMILY_GUIDE_PROMPT}\n\n{rag_context}\n\nQuestion: {question}"
    
    # Try using nano_llm as a command line tool
    try:
        import subprocess
        import tempfile
        
        print("🔄 Running nano_llm chat...")
        
        # Create a temporary file with the prompt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name
        
        try:
            # Run nano_llm chat command
            cmd = [
                'python3', '-m', 'nano_llm.chat',
                '--api', 'mlc',
                '--model', 'princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT',
                '--quantization', 'q4f16_ft',
                '--max-new-tokens', '200',
                '--prompt', prompt_file
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ nano_llm completed successfully")
                return result.stdout.strip()
            else:
                print(f"❌ nano_llm failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                raise Exception(f"nano_llm failed: {result.stderr}")
                
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(prompt_file)
            except:
                pass
        
    except Exception as e:
        # Fallback: provide basic transcript analysis
        print(f"❌ Error running nano_llm: {e}")
        return analyze_transcripts_basic(question, rag_context)

def main():
    parser = argparse.ArgumentParser(description='FamilyGuide - Family coaching assistant')
    parser.add_argument('transcript_folder', help='Path to transcript folder')
    parser.add_argument('question', help='Your question about family dynamics')
    parser.add_argument('--days', type=int, default=7, help='Days of transcripts to analyze (default: 7)')
    parser.add_argument('--max-segments', type=int, default=20, help='Max transcript segments to include (default: 20)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.transcript_folder):
        print(f"Error: Transcript folder '{args.transcript_folder}' not found")
        sys.exit(1)
    
    print("🔍 Analyzing recent family conversations...")
    rag_context = build_rag_context(args.transcript_folder, args.max_segments)
    
    print("🤖 Getting FamilyGuide response...")
    response = run_nano_llm_chat(args.question, rag_context)
    
    print("\n" + "="*60)
    print("FAMILYGUIDE RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()
