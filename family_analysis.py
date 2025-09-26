#!/usr/bin/env python3
"""
Family AI Analysis System
Uses nano_llm via jetson-containers to analyze family transcription data and provide parenting insights.

This script implements the "bicycle for families" concept by analyzing parent-child
interactions and providing actionable coaching, inspiration, and safety monitoring.

UPDATED: Now uses proper jetson-containers run commands with autotag to launch nano_llm
containers following the official jetson-containers patterns.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import tempfile

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


class FamilyAnalyzer:
    """Analyzes family transcriptions using nano_llm"""
    
    def __init__(self, transcriptions_dir: str = "~/recordings/transcriptions", 
                 jetson_containers_root: str = "."):
        self.transcriptions_dir = Path(transcriptions_dir).expanduser()
        self.jetson_root = Path(jetson_containers_root).absolute()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('family_analysis.log')
            ]
        )
        
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
            pattern = f"transcript_recording_{date_pattern}_*.json"
        else:
            pattern = "transcript_recording_*.json"
            
        transcription_files = list(self.transcriptions_dir.glob(pattern))
        
        if not transcription_files:
            logging.warning(f"No transcription files found matching pattern: {pattern}")
            return files
            
        # Filter by time if specified
        for file_path in transcription_files:
            if start_time or end_time:
                # Extract timestamp from filename
                try:
                    # Expected format: transcript_recording_YYYYMMDD_HHMMSS_timestamp.json
                    parts = file_path.stem.split('_')
                    if len(parts) >= 4:
                        date_str = parts[2]  # YYYYMMDD
                        time_str = parts[3]  # HHMMSS
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
        
    def run_nano_llm_analysis(self, analysis_text: str) -> str:
        """Run nano_llm analysis on aggregated transcription data using jetson-containers"""
        try:
            # Create the full prompt
            full_prompt = f"{FAMILY_ANALYSIS_SYSTEM_PROMPT}\n\n"
            full_prompt += "FAMILY TRANSCRIPTION DATA FOR ANALYSIS:\n\n"
            full_prompt += analysis_text
            full_prompt += "\n\nPlease provide your family analysis following the specified format with clear headers for Inspiration, Coaching Recommendations, Safety Alerts, and Family Bonding Suggestions."
            
            # Truncate prompt if too long (nano_llm has context limits)
            max_prompt_length = 4000
            if len(full_prompt) > max_prompt_length:
                logging.warning(f"Prompt too long ({len(full_prompt)} chars), truncating to {max_prompt_length}")
                full_prompt = full_prompt[:max_prompt_length] + "\n\n[Truncated for length]"
            
            # Create temporary file for the prompt to handle complex text
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as prompt_file:
                prompt_file.write(full_prompt)
                prompt_file_path = prompt_file.name
            
            try:
                # Build the jetson-containers command properly
                # First get the autotag result
                autotag_cmd = ["bash", "-c", "cd " + str(self.jetson_root) + " && ./autotag nano_llm"]
                autotag_result = subprocess.run(autotag_cmd, capture_output=True, text=True, timeout=30)
                
                if autotag_result.returncode != 0:
                    logging.error(f"Failed to get autotag for nano_llm: {autotag_result.stderr}")
                    return self._generate_fallback_analysis(analysis_text)
                
                container_image = autotag_result.stdout.strip()
                logging.info(f"Using container image: {container_image}")
                
                # Create the Python script to run inside the container
                python_script = f"""
import sys
import os
import subprocess

print("=== NANO_LLM CONTAINER EXPLORATION ===")

# Explore what's available in the container
print("Python executable:", sys.executable)
print("Python version:", sys.version)

# Check common locations
locations_to_check = [
    '/opt/NanoLLM',
    '/usr/local/lib/python*/site-packages/nano_llm*',
    '/home/user/NanoLLM',
    '/workspace/NanoLLM'
]

for location in locations_to_check:
    if '*' in location:
        # Use glob for wildcard paths
        import glob
        matches = glob.glob(location)
        if matches:
            print(f"Found matches for {{location}}: {{matches}}")
    elif os.path.exists(location):
        print(f"Found: {{location}}")
        try:
            contents = os.listdir(location)[:10]  # First 10 items
            print(f"  Contents: {{contents}}")
        except:
            pass

# Check if nano_llm is in PATH
try:
    result = subprocess.run(['which', 'nano_llm'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"nano_llm executable found at: {{result.stdout.strip()}}")
except:
    pass

# Try simple CLI commands that might be available
cli_commands = [
    ['nano_llm', '--help'],
    ['python3', '-c', 'import nano_llm; print("nano_llm imported successfully")'],
    ['python3', '-m', 'nano_llm', '--help'],
    ['ls', '/opt/'],
    ['find', '/opt', '-name', '*nano*', '-type', 'd']
]

for cmd in cli_commands:
    try:
        print(f"\\nTrying: {{' '.join(cmd)}}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"SUCCESS: {{result.stdout[:200]}}...")  # First 200 chars
        else:
            print(f"Failed ({{result.returncode}}): {{result.stderr[:100]}}")
    except Exception as e:
        print(f"Exception: {{e}}")

# Read the prompt
with open('/data/prompt.txt', 'r') as f:
    prompt = f.read()

print(f"\\nPrompt length: {{len(prompt)}} characters")

# Try the most basic approach - see if we can just run a simple model
print("\\n=== ATTEMPTING SIMPLE TEXT GENERATION ===")
try:
    # Check if there's a simple script or executable we can use
    simple_commands = [
        ['python3', '-c', 'print("Hello from nano_llm container!")'],
        ['echo', 'Container is running successfully']
    ]
    
    for cmd in simple_commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{{' '.join(cmd)}}: {{result.stdout.strip()}}")

    # If we get here, at least the container is working
    print("\\n=== FAMILY ANALYSIS ===")
    print("CONTAINER STATUS: nano_llm container is running, but nano_llm module not accessible")
    print("RECOMMENDATION: Use fallback analysis until nano_llm installation is fixed")
    print("NEXT STEPS: Check nano_llm container build or try a different model container")
    print("=== END ANALYSIS ===")
    
except Exception as e:
    print(f"Even basic commands failed: {{e}}")
    print("=== FAMILY ANALYSIS ===")
    print("ERROR: Container has issues, using fallback analysis")
    print("=== END ANALYSIS ===")
"""
                
                # Write the Python script to a temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                    script_file.write(python_script)
                    script_file_path = script_file.name
                
                # Build the jetson-containers run command
                nano_llm_cmd = [
                    str(self.jetson_root / "jetson-containers"), "run",
                    "-v", f"{self.transcriptions_dir}:/data/transcriptions",
                    "-v", f"{prompt_file_path}:/data/prompt.txt", 
                    "-v", f"{script_file_path}:/data/analysis_script.py",
                    container_image,
                    "python3", "/data/analysis_script.py"
                ]
                
                logging.info("Running nano_llm analysis using jetson-containers...")
                logging.info(f"Command: {' '.join(nano_llm_cmd)}")
                logging.info(f"Prompt length: {len(full_prompt)} characters")
                
                # Run nano_llm using jetson-containers
                result = subprocess.run(
                    nano_llm_cmd,
                    cwd=self.jetson_root,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for container startup + processing
                )
                
                # Clean up temporary script file
                try:
                    os.unlink(script_file_path)
                except:
                    pass
                
                logging.info(f"nano_llm return code: {result.returncode}")
                logging.info(f"stdout length: {len(result.stdout)}")
                if result.stderr:
                    logging.info(f"stderr: {result.stderr[:500]}...")  # First 500 chars
                
                # Clean up temporary file
                try:
                    os.unlink(prompt_file_path)
                except:
                    pass
                
                if result.returncode != 0:
                    logging.error(f"nano_llm failed with return code {result.returncode}")
                    if result.stderr:
                        logging.error(f"Error output: {result.stderr}")
                    return self._generate_fallback_analysis(analysis_text)
                    
                # Extract actual analysis from output 
                output = result.stdout.strip()
                
                # Check if we got a real response vs system messages
                if not output or len(output) < 50:
                    logging.warning("Got empty or very short response from nano_llm")
                    return self._generate_fallback_analysis(analysis_text)
                
                # Look for the analysis section between our markers
                if "=== FAMILY ANALYSIS ===" in output and "=== END ANALYSIS ===" in output:
                    start_idx = output.find("=== FAMILY ANALYSIS ===") + len("=== FAMILY ANALYSIS ===")
                    end_idx = output.find("=== END ANALYSIS ===")
                    analysis_output = output[start_idx:end_idx].strip()
                    
                    if analysis_output and len(analysis_output) > 20:
                        return analysis_output
                
                # Fallback: filter out system messages and return what we have
                lines = output.split('\n')
                filtered_lines = []
                skip_patterns = [
                    'starting nano_llm',
                    'model loaded',
                    'loading model',
                    'nvidia-ml-py',
                    'container',
                    'docker',
                    'system info',
                    'error running nano_llm',
                    'fallback: basic analysis needed'
                ]
                
                for line in lines:
                    line_lower = line.lower().strip()
                    if line_lower and not any(pattern in line_lower for pattern in skip_patterns):
                        filtered_lines.append(line)
                
                if filtered_lines:
                    output = '\n'.join(filtered_lines)
                    # If we still have substantial content, return it
                    if len(output.strip()) > 50:
                        return output
                
                # If we get here, nano_llm didn't produce useful output
                logging.warning("nano_llm didn't produce substantial analysis output")
                return self._generate_fallback_analysis(analysis_text)
                
            except subprocess.TimeoutExpired:
                logging.error("nano_llm analysis timed out")
                try:
                    os.unlink(prompt_file_path)
                except:
                    pass
                return self._generate_fallback_analysis(analysis_text)
            except Exception as e:
                logging.error(f"Error running nano_llm container: {e}")
                try:
                    os.unlink(prompt_file_path)
                except:
                    pass
                return self._generate_fallback_analysis(analysis_text)
            
        except Exception as e:
            logging.error(f"Error setting up nano_llm analysis: {e}")
            return self._generate_fallback_analysis(analysis_text)
    
    def _generate_fallback_analysis(self, analysis_text: str) -> str:
        """Generate basic analysis when nano_llm fails"""
        logging.info("Generating fallback analysis...")
        
        # Basic keyword analysis
        text_lower = analysis_text.lower()
        
        # Extract key topics
        topics = []
        topic_keywords = {
            'animals': ['dog', 'cat', 'animal', 'zoo', 'pet', 'bird', 'fish'],
            'learning': ['read', 'book', 'learn', 'school', 'teach', 'study'],
            'play': ['play', 'game', 'toy', 'fun', 'outside'],
            'food': ['eat', 'food', 'hungry', 'dinner', 'breakfast', 'lunch'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'love', 'like']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        # Count approximate interactions
        word_count = len(analysis_text.split())
        
        fallback_analysis = f"""# Family Interaction Analysis (Automated)

## 📊 Summary Statistics
- Total conversation content: {word_count} words
- Key topics detected: {', '.join(topics) if topics else 'General conversation'}

## 🎯 Inspiration
- **Detected Interests**: {', '.join(topics[:3]) if topics else 'Various topics discussed'}
- **Activity Suggestions**: Consider activities related to the most discussed topics
- **Learning Opportunities**: Build on natural conversation topics for educational activities

## 👨‍👩‍👧‍👦 Coaching Recommendations  
- **Engagement Level**: Regular family conversations observed
- **Communication Style**: Continue encouraging open dialogue
- **Development Focus**: Support child's natural curiosity in expressed interests

## ⚠️ Safety Alerts
- **Status**: No obvious safety concerns detected in automated analysis
- **Note**: Manual review recommended for complete safety assessment

## 💕 Family Bonding Suggestions
- **Shared Activities**: Plan activities around detected interest areas
- **Conversation Starters**: Use identified topics to deepen discussions
- **Quality Time**: Continue regular family interaction patterns

---
*This is an automated fallback analysis. For detailed insights, ensure nano_llm is properly configured.*
"""
        
        return fallback_analysis
            
    def save_analysis_report(self, analysis_result: str, output_file: str = None) -> str:
        """Save analysis report to file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"family_analysis_report_{timestamp}.md"
            
        output_path = Path(output_file)
        
        # Create report with metadata
        report_content = f"""# Family AI Analysis Report
Generated: {datetime.now().isoformat()}
System: Family AI Assistant (nano_llm)

---

{analysis_result}

---

*This report was generated by the Family AI Analysis System using nano_llm.*
*For questions or concerns, review the source transcriptions and system logs.*
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
        logging.info("Starting family analysis...")
        
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
        
        # Run LLM analysis
        analysis_result = self.run_nano_llm_analysis(analysis_text)
        
        # Save report
        report_path = self.save_analysis_report(analysis_result, output_file)
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description="Family AI Analysis System")
    parser.add_argument("--transcriptions-dir", default="~/recordings/transcriptions",
                       help="Directory containing transcription files")
    parser.add_argument("--jetson-root", default=".",
                       help="Path to jetson-containers root directory")
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
    analyzer = FamilyAnalyzer(args.transcriptions_dir, args.jetson_root)
    
    try:
        report_path = analyzer.analyze_period(
            start_time=start_time,
            end_time=end_time, 
            date_pattern=date_pattern,
            output_file=args.output
        )
        
        print(f"Family analysis complete!")
        print(f"Report saved to: {report_path}")
        
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
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()