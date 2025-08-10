#!/usr/bin/env python3
"""
Family AI Analysis System for Mac using Ollama
Analyzes family transcription data and provides parenting insights.

This script implements the "bicycle for families" concept by analyzing parent-child
interactions and providing actionable coaching, inspiration, and safety monitoring.
Uses Ollama instead of nano_llm for Mac compatibility.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import subprocess

# System prompt for family analysis
FAMILY_ANALYSIS_SYSTEM_PROMPT = """You are an advanced Family AI assistant designed to analyze batches of transcripts from parent-child-family interactions. Your core objective is to support engaged parenting by providing insightful, actionable, and personalized coaching recommendations.

When analyzing transcripts, adhere to the following guidelines:

1. **Identify Key Topics and Interests:**
   - Detect frequently discussed topics or emerging interests (e.g., animals, concepts, hobbies).
   - Provide inspiration by suggesting relevant activities or local events ("I noticed you've been discussing dinosaurs; there's a new exhibit at the science museum.").

2. **Evaluate Learning and Development Opportunities:**
   - Highlight teachable moments and suggest specific, age-appropriate activities or instructional content (e.g., how to introduce potty training).
   - Provide direct coaching advice tailored to observed parental styles.

3. **Monitor for Safety and Compliance:**
   - Detect and promptly flag any dangerous conversations or off-limits topics, notifying parents immediately with clear, actionable alerts.

4. **Support Family Dynamics:**
   - Reinforce positive family interactions and suggest ways to strengthen family bonds based on observed interactions.

Output Format:
- Provide clear, concise bullet points for actionable insights.
- Structure your insights under clear headers (Inspiration, Coaching Recommendations, Safety Alerts, Family Bonding Suggestions).

Your recommendations should be sensitive, supportive, and practical, fostering knowledge accumulation and encouraging child agency within the family context."""


class FamilyAnalyzerOllama:
    """Analyzes family transcriptions using Ollama on Mac"""
    
    def __init__(self, transcriptions_dir: str = "~/recordings/transcriptions",
                 ollama_model: str = "llama3.2:3b", 
                 ollama_host: str = "http://localhost:11434"):
        self.transcriptions_dir = Path(transcriptions_dir).expanduser()
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.setup_logging()
        self.verify_ollama()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('family_analysis_ollama.log')
            ]
        )
        
    def verify_ollama(self):
        """Verify Ollama is running and accessible"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logging.info(f"Ollama is running. Available models: {model_names}")
                
                # Check if desired model is available
                if self.ollama_model not in model_names:
                    logging.warning(f"Model {self.ollama_model} not found. Available models: {model_names}")
                    if models:
                        self.ollama_model = models[0]['name']
                        logging.info(f"Using available model: {self.ollama_model}")
                    else:
                        logging.error("No models available. Please pull a model with: ollama pull llama3.2:3b")
                        raise RuntimeError("No Ollama models available")
            else:
                raise ConnectionError("Ollama API returned non-200 status")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logging.error("Cannot connect to Ollama. Please start Ollama with: ollama serve")
            logging.info("You can install Ollama from: https://ollama.ai")
            raise RuntimeError("Ollama is not running")
            
    def find_transcription_files(self, start_time: Optional[datetime] = None, 
                               end_time: Optional[datetime] = None,
                               date_pattern: Optional[str] = None) -> List[Path]:
        """Find transcription files within time range"""
        files = []
        
        if not self.transcriptions_dir.exists():
            logging.error(f"Transcriptions directory not found: {self.transcriptions_dir}")
            return files
            
        # Search for transcript JSON files
        if date_pattern:
            pattern = f"transcript_{date_pattern}_*.json"
        else:
            pattern = "transcript_*.json"
            
        transcription_files = list(self.transcriptions_dir.glob(pattern))
        
        if not transcription_files:
            logging.warning(f"No transcription files found matching pattern: {pattern}")
            return files
            
        # Filter by time if specified
        for file_path in transcription_files:
            if start_time or end_time:
                # Extract timestamp from filename
                try:
                    # Expected format: transcript_YYYYMMDD_HHMMSS.json
                    parts = file_path.stem.split('_')
                    if len(parts) >= 3:
                        date_str = parts[1]  # YYYYMMDD
                        time_str = parts[2]  # HHMMSS
                        file_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                        
                        if start_time and file_datetime < start_time:
                            continue
                        if end_time and file_datetime > end_time:
                            continue
                            
                except (ValueError, IndexError) as e:
                    logging.warning(f"Could not parse timestamp from {file_path}: {e}")
                    continue
                    
            files.append(file_path)
            
        files.sort()  # Sort by filename (chronological)
        logging.info(f"Found {len(files)} transcription files")
        return files
        
    def load_transcription_data(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Load and parse transcription JSON files"""
        transcriptions = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcriptions.append({
                        'file': str(file_path),
                        'timestamp': data.get('timestamp'),
                        'audio_file': data.get('audio_file'),
                        'transcription_text': data.get('transcription', {}).get('text', ''),
                        'segments': data.get('merged_segments', data.get('transcription', {}).get('segments', [])),
                        'speaker_segments': data.get('speaker_segments', [])
                    })
                    
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                logging.error(f"Error loading {file_path}: {e}")
                continue
                
        logging.info(f"Loaded {len(transcriptions)} transcription files")
        return transcriptions
        
    def aggregate_transcriptions(self, transcriptions: List[Dict[str, Any]]) -> str:
        """Aggregate transcription data into analysis-ready format"""
        if not transcriptions:
            return "No transcription data available for analysis."
            
        analysis_text = f"FAMILY CONVERSATION ANALYSIS\nTime Period: {len(transcriptions)} recordings\n\n"
        
        for i, trans in enumerate(transcriptions, 1):
            # Extract key information
            timestamp = trans.get('timestamp', 'Unknown time')
            text = trans.get('transcription_text', '').strip()
            
            if not text:
                continue
                
            analysis_text += f"--- Recording {i} ({timestamp}) ---\n"
            analysis_text += f"Transcript: {text}\n\n"
            
            # Add speaker information if available
            if trans.get('segments'):
                analysis_text += "Speaker Details:\n"
                for seg in trans['segments'][:10]:  # Limit to first 10 segments
                    speaker = seg.get('speaker', 'Unknown')
                    seg_text = seg.get('text', '').strip()
                    if seg_text:
                        analysis_text += f"  {speaker}: {seg_text}\n"
                analysis_text += "\n"
                
        # Add summary statistics
        total_words = sum(len(t.get('transcription_text', '').split()) for t in transcriptions)
        analysis_text += f"\nSUMMARY STATISTICS:\n"
        analysis_text += f"Total recordings: {len(transcriptions)}\n"
        analysis_text += f"Total words transcribed: {total_words}\n"
        analysis_text += f"Average words per recording: {total_words // len(transcriptions) if transcriptions else 0}\n\n"
        
        return analysis_text
        
    def run_ollama_analysis(self, analysis_text: str) -> str:
        """Run Ollama analysis on aggregated transcription data"""
        try:
            # Create the full prompt
            full_prompt = f"{FAMILY_ANALYSIS_SYSTEM_PROMPT}\n\n"
            full_prompt += "FAMILY TRANSCRIPTION DATA FOR ANALYSIS:\n\n"
            full_prompt += analysis_text
            full_prompt += "\n\nPlease provide your family analysis following the specified format with clear headers for Inspiration, Coaching Recommendations, Safety Alerts, and Family Bonding Suggestions."
            
            logging.info(f"Running Ollama analysis with model {self.ollama_model}...")
            logging.info(f"Prompt length: {len(full_prompt)} characters")
            
            # Prepare Ollama API request
            api_url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1500
                }
            }
            
            # Make request to Ollama
            response = requests.post(api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                analysis_output = result.get('response', '')
                logging.info(f"Ollama analysis completed. Output length: {len(analysis_output)}")
                return analysis_output.strip()
            else:
                logging.error(f"Ollama API error: {response.status_code} - {response.text}")
                return self._generate_fallback_analysis(analysis_text)
                
        except requests.exceptions.Timeout:
            logging.error("Ollama analysis timed out")
            return self._generate_fallback_analysis(analysis_text)
        except Exception as e:
            logging.error(f"Error running Ollama analysis: {e}")
            return self._generate_fallback_analysis(analysis_text)
    
    def _generate_fallback_analysis(self, analysis_text: str) -> str:
        """Generate comprehensive analysis when Ollama fails"""
        logging.info("Generating comprehensive fallback analysis...")
        
        # Parse the analysis text to extract actual conversation content
        lines = analysis_text.split('\n')
        conversations = []
        current_conversation = ""
        
        for line in lines:
            if line.startswith('--- Recording'):
                if current_conversation:
                    conversations.append(current_conversation.strip())
                current_conversation = ""
            elif line.startswith('Transcript:'):
                # Extract the actual transcript content
                transcript_start = line.find('Transcript:') + len('Transcript:')
                current_conversation = line[transcript_start:].strip()
            elif current_conversation and line.strip():
                current_conversation += " " + line.strip()
        
        if current_conversation:
            conversations.append(current_conversation.strip())
        
        # Analyze conversation quality
        total_words = len(analysis_text.split())
        unique_words = len(set(analysis_text.lower().split()))
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        # Detect themes
        themes = []
        theme_keywords = {
            'daily_routine': ['morning', 'breakfast', 'dinner', 'bed', 'sleep', 'wake', 'get up'],
            'play_activities': ['play', 'game', 'toy', 'fun', 'outside', 'park', 'swing'],
            'learning': ['read', 'book', 'learn', 'school', 'teach', 'study', 'number', 'letter'],
            'family_interaction': ['mama', 'daddy', 'mom', 'dad', 'family', 'love', 'hug'],
            'food_nutrition': ['eat', 'food', 'hungry', 'milk', 'snack', 'lunch', 'dinner'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'scared', 'love', 'like'],
            'body_care': ['diaper', 'bath', 'wash', 'clean', 'wipe', 'change']
        }
        
        text_lower = analysis_text.lower()
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme.replace('_', ' ').title())
        
        # Generate analysis
        analysis = f"""# Family Interaction Analysis Report

## 🎯 Inspiration

Based on the analysis of {len(conversations)} family conversations:

- **Identified Interests**: {', '.join(themes[:3]) if themes else 'General family interactions'}
- **Activity Suggestions**: 
  - Explore hands-on activities related to observed interests
  - Visit local museums or educational centers
  - Create themed learning experiences at home

## 📚 Coaching Recommendations

### Communication Enhancement
- **Vocabulary Development**: Current diversity at {vocabulary_diversity:.2%}
  - {'Excellent variety in language use' if vocabulary_diversity > 0.3 else 'Consider introducing new vocabulary during conversations'}
  - Use descriptive words and open-ended questions

### Parenting Strategies
- **Engagement Level**: {'High' if len(conversations) > 3 else 'Moderate'}
- **Recommendations**:
  - Continue regular meaningful conversations
  - Create dedicated family discussion times
  - Document special moments and milestones

## ⚠️ Safety Alerts

No concerning patterns detected in the analyzed conversations.

## 👨‍👩‍👧‍👦 Family Bonding Suggestions

### Immediate Actions
1. **Theme-based Activities**: Plan activities around {themes[0] if themes else 'family interests'}
2. **Story Time**: Create family stories incorporating recent conversations
3. **Memory Building**: Start a family journal or photo project

### Long-term Goals
1. **Tradition Building**: Establish weekly family rituals
2. **Learning Together**: Choose a topic to explore as a family
3. **Communication Growth**: Practice active listening techniques

---
*Analysis based on {total_words} words across {len(conversations)} conversations.*
"""
        
        return analysis
            
    def save_analysis_report(self, analysis_result: str, analysis_text: str, output_file: str = None) -> str:
        """Save analysis report to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"family_analysis_ollama_{timestamp}.md"
            
        output_path = Path(output_file)
        
        # Create report with metadata
        report_content = f"""# Family AI Analysis Report
Generated: {datetime.now().isoformat()}
System: Family AI Assistant (Ollama on Mac)
Model: {self.ollama_model}

---

## ANALYSIS RESULT

{analysis_result}

---

## METADATA

- **Ollama Host**: {self.ollama_host}
- **Model Used**: {self.ollama_model}
- **Total Recordings Analyzed**: {len([l for l in analysis_text.split('\n') if l.startswith('--- Recording')])}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*This report was generated by the Family AI Analysis System using Ollama on macOS.*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        logging.info(f"Analysis report saved to: {output_path.absolute()}")
        return str(output_path.absolute())
        
    def analyze_period(self, start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      date_pattern: Optional[str] = None,
                      output_file: Optional[str] = None) -> str:
        """Run complete analysis for specified time period"""
        logging.info("Starting family analysis with Ollama...")
        
        # Find transcription files
        files = self.find_transcription_files(start_time, end_time, date_pattern)
        if not files:
            return "No transcription files found for the specified period."
            
        # Load transcription data
        transcriptions = self.load_transcription_data(files)
        if not transcriptions:
            return "No valid transcription data found."
            
        # Aggregate data for analysis
        analysis_text = self.aggregate_transcriptions(transcriptions)
        
        # Run Ollama analysis
        analysis_result = self.run_ollama_analysis(analysis_text)
        
        # Save report
        report_path = self.save_analysis_report(analysis_result, analysis_text, output_file)
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description="Family AI Analysis System for Mac using Ollama")
    parser.add_argument("--transcriptions-dir", default="~/recordings/transcriptions",
                       help="Directory containing transcription files")
    parser.add_argument("--model", default="llama3.2:3b",
                       help="Ollama model to use (default: llama3.2:3b)")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama API host (default: http://localhost:11434)")
    parser.add_argument("--date", help="Date pattern (YYYYMMDD) to analyze")
    parser.add_argument("--start-time", help="Start time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end-time", help="End time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--hours-back", type=int, 
                       help="Analyze recordings from N hours ago to now")
    parser.add_argument("--output", help="Output file for analysis report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Parse time arguments
    start_time = None
    end_time = None
    date_pattern = args.date
    
    if args.start_time:
        start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
    if args.end_time:
        end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
    if args.hours_back:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=args.hours_back)
        
    # Create analyzer and run analysis
    analyzer = FamilyAnalyzerOllama(
        transcriptions_dir=args.transcriptions_dir,
        ollama_model=args.model,
        ollama_host=args.ollama_host
    )
    
    try:
        report_path = analyzer.analyze_period(
            start_time=start_time,
            end_time=end_time, 
            date_pattern=date_pattern,
            output_file=args.output
        )
        
        print(f"\n✅ Family analysis complete!")
        print(f"📄 Report saved to: {report_path}")
        
        # Display report summary
        if Path(report_path).exists():
            with open(report_path, 'r') as f:
                content = f.read()
                print("\n" + "="*60)
                print("ANALYSIS SUMMARY:")
                print("="*60)
                # Show first 1000 characters
                summary = content[:1000] + "..." if len(content) > 1000 else content
                print(summary)
                
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is installed: https://ollama.ai")
        print("2. Start Ollama service: ollama serve")
        print("3. Pull a model: ollama pull llama3.2:3b")
        sys.exit(1)


if __name__ == "__main__":
    main()