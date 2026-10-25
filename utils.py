# utils.py — Session logging and analytics
import time
import csv
import os

LOGFILE = "drowsiness_log.csv"


class SessionLogger:
    """Tracks session events and writes summaries."""

    def __init__(self, logfile=LOGFILE):
        self.logfile = logfile
        self.events = []
        self.session_start = time.time()
        self.was_alert = False
        self.was_warning = False

    def log_event(self, message):
        """Append a timestamped event."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.events.append((ts, message))

        header_needed = not os.path.exists(self.logfile)
        try:
            with open(self.logfile, "a", newline="") as f:
                writer = csv.writer(f)
                if header_needed:
                    writer.writerow(["timestamp", "event"])
                writer.writerow([ts, message])
        except Exception:
            pass

    def write_summary(self, detector=None):
        """Write session summary to log."""
        duration = time.time() - self.session_start
        mins = int(duration // 60)
        secs = int(duration % 60)

        summary = f"Session ended — Duration: {mins}m {secs}s"
        if detector:
            summary += (f", Alerts: {detector.total_drowsy_events}"
                        f", Yawns: {detector.total_yawns}"
                        f", Avg EAR: {detector.get_avg_ear():.3f}")

        self.log_event(summary)
        print(f"\n{summary}")
