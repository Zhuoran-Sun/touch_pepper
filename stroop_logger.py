#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone ROS 2 Stroop TRUE/FALSE logger.

This subscribes to:
  - /stroop/answer_detail   std_msgs/msg/String   preferred detailed log
  - /stroop/answer          std_msgs/msg/Bool     fallback simple TRUE/FALSE log
  - /stroop/result          std_msgs/msg/Bool     fallback simple TRUE/FALSE log
  - /stroop/trial           std_msgs/msg/String   current trial info
  - /stroop/finished        std_msgs/msg/Bool     closes/flushed log when task ends

Run:
  source ~/ros2_ws/install/setup.bash
  python3 stroop_response_logger.py

Optional:
  python3 stroop_response_logger.py --output-dir /home/gvlab/ran/stroop_logs --run-id p01
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
try:
    from rclpy.executors import ExternalShutdownException
except Exception:
    ExternalShutdownException = Exception
from std_msgs.msg import Bool, String


RESPONSE_COLUMNS = [
    "local_time",
    "ros_time_sec",
    "trial_id",
    "answer",
    "is_correct",
    "source",
    "response_time_ms",
    "word",
    "ink",
    "ground_truth",
    "phase",
    "screen",
    "task",
    "topic",
    "raw_message",
]

TRIAL_COLUMNS = [
    "local_time",
    "ros_time_sec",
    "trial_id",
    "screen",
    "phase",
    "task",
    "duration_ms",
    "word",
    "ink",
    "ground_truth",
    "raw_message",
]


def parse_semicolon_kv(text: str) -> Dict[str, str]:
    """Parse strings like 'id=1;answer=true;word=red' into a dict."""
    out: Dict[str, str] = {}
    for part in str(text).split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def bool_to_text(value: bool) -> str:
    return "true" if bool(value) else "false"


class StroopResponseLogger(Node):
    def __init__(self, output_dir: str, run_id: Optional[str]):
        super().__init__("stroop_response_logger")

        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # output_dir = os.path.expanduser(output_dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "stroop_logs")
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.response_path = os.path.join(output_dir, f"stroop_responses_{run_id}.csv")
        self.trial_path = os.path.join(output_dir, f"stroop_trials_{run_id}.csv")

        self.response_file = open(self.response_path, "w", newline="", encoding="utf-8")
        self.trial_file = open(self.trial_path, "w", newline="", encoding="utf-8")

        self.response_writer = csv.DictWriter(self.response_file, fieldnames=RESPONSE_COLUMNS)
        self.trial_writer = csv.DictWriter(self.trial_file, fieldnames=TRIAL_COLUMNS)
        self.response_writer.writeheader()
        self.trial_writer.writeheader()
        self.response_file.flush()
        self.trial_file.flush()

        self.current_trial: Dict[str, str] = {}
        self.simple_counter = 0
        self.closed = False

        # Detailed topic from your Stroop script. This is the best source.
        self.create_subscription(String, "/stroop/answer_detail", self.on_answer_detail, 10)

        # Trial topic lets us recover trial metadata if only simple Bool messages are available.
        self.create_subscription(String, "/stroop/trial", self.on_trial, 10)

        # Simple fallback topics.
        self.create_subscription(Bool, "/stroop/answer", lambda msg: self.on_simple_bool(msg, "/stroop/answer"), 10)
        self.create_subscription(Bool, "/stroop/result", lambda msg: self.on_simple_bool(msg, "/stroop/result"), 10)

        self.create_subscription(Bool, "/stroop/finished", self.on_finished, 10)

        self.get_logger().info(f"Logging Stroop responses to: {self.response_path}")
        self.get_logger().info(f"Logging Stroop trials to:     {self.trial_path}")
        self.get_logger().info("Waiting for /stroop/answer_detail, /stroop/trial, /stroop/answer, /stroop/result ...")

    def now_fields(self) -> Dict[str, str]:
        now_msg = self.get_clock().now().to_msg()
        ros_sec = now_msg.sec + now_msg.nanosec / 1e9
        return {
            "local_time": datetime.now().isoformat(timespec="milliseconds"),
            "ros_time_sec": f"{ros_sec:.9f}",
        }

    def on_trial(self, msg: String) -> None:
        data = parse_semicolon_kv(msg.data)
        self.current_trial = data

        row = {col: "" for col in TRIAL_COLUMNS}
        row.update(self.now_fields())
        row.update({
            "trial_id": data.get("id", ""),
            "screen": data.get("screen", ""),
            "phase": data.get("phase", ""),
            "task": data.get("task", ""),
            "duration_ms": data.get("duration_ms", ""),
            "word": data.get("word", ""),
            "ink": data.get("ink", ""),
            "ground_truth": data.get("ground_truth", ""),
            "raw_message": msg.data,
        })
        self.trial_writer.writerow(row)
        self.trial_file.flush()

    def on_answer_detail(self, msg: String) -> None:
        data = parse_semicolon_kv(msg.data)
        answer = data.get("answer", "").lower()

        row = {col: "" for col in RESPONSE_COLUMNS}
        row.update(self.now_fields())
        row.update({
            "trial_id": data.get("id", ""),
            "answer": answer,
            # In your current Stroop script, answer=true means the manager marked it correct.
            # answer=false means wrong or timeout.
            "is_correct": answer,
            "source": data.get("source", ""),
            "response_time_ms": data.get("response_time_ms", ""),
            "word": data.get("word", ""),
            "ink": data.get("ink", ""),
            "ground_truth": data.get("ground_truth", ""),
            "phase": data.get("phase", ""),
            "screen": data.get("screen", ""),
            "task": data.get("task", ""),
            "topic": "/stroop/answer_detail",
            "raw_message": msg.data,
        })
        self.response_writer.writerow(row)
        self.response_file.flush()
        self.get_logger().info(
            f"Logged detail: trial={row['trial_id']} answer={row['answer']} "
            f"phase={row['phase']} rt={row['response_time_ms']}ms"
        )

    def on_simple_bool(self, msg: Bool, topic: str) -> None:
        # Avoid double-logging if you are using /stroop/answer_detail.
        # But keep this useful for older scripts that only publish Bool.
        self.simple_counter += 1
        answer = bool_to_text(msg.data)
        t = self.current_trial

        row = {col: "" for col in RESPONSE_COLUMNS}
        row.update(self.now_fields())
        row.update({
            "trial_id": t.get("id", str(self.simple_counter)),
            "answer": answer,
            "is_correct": answer,
            "source": "simple_bool",
            "response_time_ms": "",
            "word": t.get("word", ""),
            "ink": t.get("ink", ""),
            "ground_truth": t.get("ground_truth", ""),
            "phase": t.get("phase", ""),
            "screen": t.get("screen", ""),
            "task": t.get("task", ""),
            "topic": topic,
            "raw_message": answer,
        })
        self.response_writer.writerow(row)
        self.response_file.flush()
        self.get_logger().info(f"Logged simple Bool from {topic}: trial={row['trial_id']} answer={answer}")

    def on_finished(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info("Received /stroop/finished=True. Log files flushed.")
            self.response_file.flush()
            self.trial_file.flush()

    def close_files(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.response_file.flush()
            self.trial_file.flush()
            self.response_file.close()
            self.trial_file.close()
            self.get_logger().info(f"Saved: {self.response_path}")
            self.get_logger().info(f"Saved: {self.trial_path}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone ROS2 logger for Stroop TRUE/FALSE responses.")
    parser.add_argument("--output-dir", default="~/stroop_logs", help="Folder for CSV logs. Default: ~/stroop_logs")
    parser.add_argument("--run-id", default=None, help="Optional run name, e.g. participant_01")
    args = parser.parse_args()

    rclpy.init()
    node = StroopResponseLogger(args.output_dir, args.run_id)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\nStopping logger...")
    finally:
        node.close_files()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()