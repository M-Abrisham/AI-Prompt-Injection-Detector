#!/usr/bin/env python3
"""Simple web dashboard for Na0S approval history.

Displays approval/deployment history, statistics, and timeline charts.
Uses only stdlib (http.server, json, pathlib) - no external dependencies.

Usage:
    python scripts/approval_dashboard.py --port 8080
    Then open http://localhost:8080 in browser

Features:
- Recent 30 approvals table (date, action, status, result)
- Stats summary (total, success rate, avg execution time)
- Timeline chart (approvals per day for last 30 days)
- Mobile-responsive HTML (works in Safari on iOS)
- JSON API endpoints for integration
"""

import json
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from na0s.agents.approval_history import ApprovalHistoryManager

logger = logging.getLogger(__name__)

# Global history manager (initialized in main)
HISTORY_MANAGER = None
DATA_DIR = "data"


def get_timeline_data(days: int = 30, history_manager=None) -> Dict[str, int]:
    """Generate timeline data: approvals per day for last N days.

    Args:
        days: Number of days to include
        history_manager: Optional ApprovalHistoryManager instance (uses global if None)

    Returns:
        Dict mapping date (YYYY-MM-DD) to approval count
    """
    mgr = history_manager or HISTORY_MANAGER
    if not mgr:
        return {}
    records = mgr.get_recent(days=days)

    timeline = {}
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        timeline[date] = 0

    for record in records:
        try:
            timestamp = record.get("timestamp", "")
            if timestamp:
                date = timestamp.split("T")[0]
                if date in timeline:
                    timeline[date] += 1
        except Exception:
            pass

    return dict(sorted(timeline.items()))


def render_html_dashboard(stats: Dict[str, Any], recent: List[Dict[str, Any]], history_manager=None) -> str:
    """Render HTML dashboard page.

    Args:
        stats: Stats dict from ApprovalHistoryManager.get_stats()
        recent: Recent approvals list
        history_manager: Optional ApprovalHistoryManager instance for timeline data

    Returns:
        HTML string
    """
    timeline = get_timeline_data(days=30, history_manager=history_manager)
    timeline_dates = list(timeline.keys())
    timeline_counts = list(timeline.values())

    # Build action type breakdown
    action_stats = stats.get("by_action_type", {})
    action_labels = list(action_stats.keys())
    action_counts = [action_stats[k].get("count", 0) for k in action_labels]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Na0S Approval Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a1a;
            font-size: 28px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #666;
            font-size: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #666;
            font-size: 14px;
        }}
        .chart {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .bar-chart {{
            display: flex;
            align-items: flex-end;
            gap: 10px;
            height: 200px;
        }}
        .bar {{
            flex: 1;
            background: #2196F3;
            border-radius: 4px 4px 0 0;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            min-height: 20px;
            position: relative;
        }}
        .bar-label {{
            position: absolute;
            bottom: -25px;
            font-size: 12px;
            text-align: center;
            white-space: nowrap;
            width: 100%;
        }}
        .bar-value {{
            color: white;
            font-size: 12px;
            font-weight: bold;
            padding-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background: #f9f9f9;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid #ddd;
            font-size: 14px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 14px;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .status-approved {{
            color: #4caf50;
            font-weight: 500;
        }}
        .status-rejected {{
            color: #f44336;
            font-weight: 500;
        }}
        .status-failed {{
            color: #ff9800;
            font-weight: 500;
        }}
        .action-type {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        .timestamp {{
            color: #999;
            font-size: 13px;
        }}
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            table {{
                font-size: 12px;
            }}
            th, td {{
                padding: 8px;
            }}
            .stat-value {{
                font-size: 24px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Na0S Approval Dashboard</h1>
            <p class="subtitle">Deployment & Quarantine Decision History</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get("total_approvals", 0)}</div>
                <div class="stat-label">Total Approvals (30d)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("success_rate", 0):.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("successful_approvals", 0)}</div>
                <div class="stat-label">Successful Actions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("avg_execution_time_seconds", 0):.1f}s</div>
                <div class="stat-label">Avg Execution Time</div>
            </div>
        </div>

        <div class="chart">
            <div class="chart-title">Approvals per Day (Last 30 Days)</div>
            <div class="bar-chart">
"""

    if timeline_counts:
        max_count = max(timeline_counts) or 1
        for date, count in zip(timeline_dates, timeline_counts):
            height_percent = (count / max_count * 100) if max_count > 0 else 0
            html += f"""
                <div class="bar" style="height: {height_percent}%;">
                    <div class="bar-value">{count}</div>
                    <div class="bar-label">{date[-5:]}</div>
                </div>
"""

    html += """
            </div>
        </div>

        <div class="chart">
            <div class="chart-title">Actions by Type (30d)</div>
            <div class="bar-chart">
"""

    if action_counts:
        max_action = max(action_counts) or 1
        for label, count in zip(action_labels, action_counts):
            height_percent = (count / max_action * 100) if max_action > 0 else 0
            html += f"""
                <div class="bar" style="height: {height_percent}%;">
                    <div class="bar-value">{count}</div>
                    <div class="bar-label">{label}</div>
                </div>
"""

    html += """
            </div>
        </div>

        <div class="chart">
            <div class="chart-title">Recent Approvals (Last 30)</div>
            <table>
                <thead>
                    <tr>
                        <th>Date & Time</th>
                        <th>Action</th>
                        <th>Status</th>
                        <th>Result</th>
                        <th>Time (s)</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
"""

    if recent:
        for record in recent[:30]:
            timestamp = record.get("timestamp", "")
            action = record.get("action_type", "unknown")
            status = record.get("status", "unknown")
            result = record.get("execution_result", "-")
            exec_time = record.get("execution_time_seconds", 0)
            reason = record.get("reason", "")

            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = timestamp

            # Status class
            status_class = f"status-{status}"

            # Format execution time
            exec_time_str = f"{exec_time:.1f}" if exec_time else "0"

            html += f"""
                    <tr>
                        <td><span class="timestamp">{time_str}</span></td>
                        <td><span class="action-type">{action}</span></td>
                        <td><span class="{status_class}">{status}</span></td>
                        <td>{result}</td>
                        <td>{exec_time_str}</td>
                        <td>{reason[:50] if reason else "-"}</td>
                    </tr>
"""
    else:
        html += """
                    <tr>
                        <td colspan="6" style="text-align: center; color: #999;">No approvals recorded yet</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    return html


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard server."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        try:
            if path == "/":
                # Main dashboard page
                stats = HISTORY_MANAGER.get_stats(days=30)
                recent = HISTORY_MANAGER.get_recent(days=30)
                html = render_html_dashboard(stats, recent)

                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))

            elif path == "/api/approvals":
                # JSON API: recent approvals
                days = int(query_params.get("days", ["30"])[0])
                action_type = query_params.get("action_type", [None])[0]
                records = HISTORY_MANAGER.get_recent(days=days, action_type=action_type)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(records).encode("utf-8"))

            elif path == "/api/stats":
                # JSON API: statistics
                days = int(query_params.get("days", ["30"])[0])
                stats = HISTORY_MANAGER.get_stats(days=days)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode("utf-8"))

            elif path == "/api/deployments":
                # JSON API: deployment history
                deployments = HISTORY_MANAGER.get_deployment_history()

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(deployments).encode("utf-8"))

            elif path == "/api/quarantine":
                # JSON API: quarantine history
                quarantine = HISTORY_MANAGER.get_quarantine_history()

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(quarantine).encode("utf-8"))

            else:
                # 404
                self.send_response(404)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Not Found")

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.send_response(500)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Internal Server Error: {e}".encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default logging."""
        logger.debug(format % args)


def main():
    """Start dashboard server."""
    global HISTORY_MANAGER, DATA_DIR

    parser = argparse.ArgumentParser(description="Na0S Approval Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080)")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default localhost)")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    args = parser.parse_args()

    # Initialize history manager
    DATA_DIR = args.data_dir
    HISTORY_MANAGER = ApprovalHistoryManager(data_dir=DATA_DIR)

    # Start server
    server = HTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"

    print(f"\n{'='*50}")
    print(f"Na0S Approval Dashboard")
    print(f"{'='*50}")
    print(f"Server running at: {url}")
    print(f"Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
