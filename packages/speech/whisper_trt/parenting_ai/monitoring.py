"""
Simple Resource Monitoring for Parenting AI SMS System
"""

import json
import logging
import psutil
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, jsonify, render_template_string

from .utils import get_system_info, load_config

class MonitoringService:
    """Simple monitoring service for system resources and service status"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_config = config.get('monitoring', {})
        self.logger = logging.getLogger("parenting_ai.monitoring")
        
        # Initialize Flask app if monitoring is enabled
        self.app = None
        if self.monitoring_config.get('enabled', True):
            self.app = self._create_flask_app()
    
    def _create_flask_app(self) -> Flask:
        """Create Flask app for monitoring dashboard"""
        app = Flask(__name__)
        
        @app.route('/status')
        def status():
            """Get system status as JSON"""
            return jsonify(self.get_system_status())
        
        @app.route('/')
        def dashboard():
            """Simple HTML dashboard"""
            return render_template_string(self._get_dashboard_template())
        
        return app
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Basic system info
            system_info = get_system_info()
            
            # Service status
            service_status = self._check_services()
            
            # Process information
            process_info = self._get_process_info()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system': system_info,
                'services': service_status,
                'processes': process_info,
                'status': 'healthy' if self._is_system_healthy() else 'warning'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'error'
            }
    
    def _check_services(self) -> Dict[str, Any]:
        """Check status of key services"""
        services = {}
        
        # Check Ollama
        try:
            result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/tags'], 
                                  capture_output=True, text=True, timeout=5)
            services['ollama'] = {
                'status': 'running' if result.returncode == 0 else 'stopped',
                'response_time': 'N/A'
            }
        except Exception:
            services['ollama'] = {'status': 'unknown', 'error': 'Connection failed'}
        
        # Check if SMS server is running (check for port 5000)
        try:
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
            sms_running = ':5000' in result.stdout
            services['sms_server'] = {
                'status': 'running' if sms_running else 'stopped',
                'port': 5000
            }
        except Exception:
            services['sms_server'] = {'status': 'unknown', 'error': 'Check failed'}
        
        return services
    
    def _get_process_info(self) -> Dict[str, Any]:
        """Get information about relevant processes"""
        processes = {}
        
        try:
            # Find relevant processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    name = proc.info['name'].lower()
                    if any(keyword in name for keyword in ['ollama', 'python', 'whisper']):
                        processes[proc.info['name']] = {
                            'pid': proc.info['pid'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            processes['error'] = str(e)
        
        return processes
    
    def _is_system_healthy(self) -> bool:
        """Check if system is in healthy state"""
        try:
            # Check memory usage
            memory_info = get_system_info()
            memory_used_percent = memory_info.get('memory_used_percent', 0)
            
            # Check load average
            load_1min = memory_info.get('load_1min', 0)
            
            # System is healthy if:
            # - Memory usage < 90%
            # - Load average < 4.0 (for 4-core system)
            return memory_used_percent < 90 and load_1min < 4.0
            
        except Exception:
            return False
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Parenting AI System Status</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .healthy { background: #d4edda; color: #155724; }
        .warning { background: #fff3cd; color: #856404; }
        .error { background: #f8d7da; color: #721c24; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 12px; color: #666; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Parenting AI System Status</h1>
        <p>Last updated: <span id="timestamp">{{ timestamp }}</span></p>
        
        <div class="status {{ status_class }}">
            <strong>System Status: {{ status }}</strong>
        </div>
        
        <h2>📊 System Resources</h2>
        <div class="metric">
            <div class="metric-value">{{ system.memory_used_percent|round(1) }}%</div>
            <div class="metric-label">Memory Used</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ system.load_1min|round(2) }}</div>
            <div class="metric-label">Load (1min)</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ system.memory_available_mb|round(0) }}MB</div>
            <div class="metric-label">Available Memory</div>
        </div>
        
        <h2>🔧 Services</h2>
        {% for service, info in services.items() %}
        <div class="status {{ 'healthy' if info.status == 'running' else 'warning' }}">
            <strong>{{ service|title }}:</strong> {{ info.status }}
            {% if info.get('port') %} (Port {{ info.port }}){% endif %}
        </div>
        {% endfor %}
        
        <h2>⚙️ Processes</h2>
        {% for name, info in processes.items() %}
        <div class="metric">
            <div class="metric-value">{{ info.cpu_percent|round(1) }}%</div>
            <div class="metric-label">{{ name }} CPU</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ info.memory_mb|round(0) }}MB</div>
            <div class="metric-label">{{ name }} Memory</div>
        </div>
        {% endfor %}
        
        <h2>📝 Raw Status</h2>
        <pre>{{ status_json }}</pre>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
        """
    
    def run_dashboard(self, host: str = '0.0.0.0', port: int = None) -> None:
        """Run the monitoring dashboard"""
        if not self.app:
            self.logger.error("Monitoring dashboard not enabled")
            return
        
        port = port or self.monitoring_config.get('port', 5001)
        
        self.logger.info(f"Starting monitoring dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=False)

def main():
    """Main function for running monitoring dashboard"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parenting AI Monitoring Dashboard")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Create and run monitoring service
    monitoring = MonitoringService(config)
    monitoring.run_dashboard(host=args.host, port=args.port)



