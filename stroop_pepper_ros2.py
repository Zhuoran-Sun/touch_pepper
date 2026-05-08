#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Stroop game controller for Pepper through naoqi_driver2.

This version removes the custom robot-arm services used by the old Stroop code and
replaces them with Pepper arm/hand commands published to naoqi_driver2's
/joint_angles topic.

Expected driver-side interfaces:
  - /joint_angles: naoqi_bridge_msgs/msg/JointAnglesWithSpeed
  - /speech: std_msgs/msg/String, optional

Run naoqi_driver2 first, e.g.:
  ros2 launch naoqi_driver naoqi_driver.launch.py nao_ip:=<PEPPER_IP>

Recommended before starting the experiment:
  ssh nao@<PEPPER_IP>
  qicli call ALAutonomousLife.setState disabled
  qicli call ALMotion.wakeUp
"""

import random
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
try:
    from rclpy.signals import SignalHandlerOptions
except Exception:
    SignalHandlerOptions = None
from rclpy.node import Node
from std_msgs.msg import Bool, String
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed


DEFAULT_COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
STUDY_COLORS = ["black", "blue", "red", "green", "orange", "pink", "purple", "brown"]

RIGHT_ARM_JOINTS = [
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
    "RHand",
]
RIGHT_HAND = "RHand"

LEFT_ARM_JOINTS = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
]
LEFT_HAND = "LHand"

# Right-arm table placement sequence copied from your Python 2 NAOqi test script.
# Tuple format: (joint angles for RIGHT_ARM_JOINTS, speed_fraction, hold_seconds)
RIGHT_TABLE_POSE_SEQUENCE = [
    ([1.6060779094696045, -0.008726646192371845, 1.745670199394226, 1.5620696544647217, -0.1825878620147705, 1.0], 1.0, 0.8),
    ([1.6060779094696045, -0.008726646192371845, 1.7303303480148315, 1.5620696544647217, -1.6951122283935547, 1.0], 1.0, 0.8),
    ([0.6473398208618164, -0.008726646192371845, 1.7272623777389526, 1.2624661922454834, -1.8238691091537476, 1.0], 1.0, 0.8),
    ([0.6519417762756348, -0.008726646192371845, 1.728796362876892, 0.4709320068359375, -1.8238691091537476, 1.0], 1.0, 0.8),
    # ([0.8912427425384521, -0.11504864692687988, 1.5953400135040283, 1.1673595905303955, -1.5509161949157715, 1.0], 1.0, 0.8),
    # ([1.118272066116333, -0.1242525577545166, 1.604543924331665, 1.1305439472198486, -1.2671260833740234, 1.0], 1.0, 0.8),
]

# Conservative left-arm placeholder. Tune on hardware before use.
LEFT_TABLE_POSE_SEQUENCE: List[Tuple[List[float], float, float]] = [
    ([0.0, 0.25, -0.30, -0.30, -1.57, 1.0], 0.2, 1.0),
]

# Final arm pose after reversing the table-placement sequence.
# This approximates a neutral Pepper standing arm configuration when using only
# /joint_angles. If your setup exposes ALRobotPosture through a ROS service,
# using StandInit there would be more exact.
RIGHT_STAND_ARM_POSE = [1.45, -0.10, 1.20, 0.50, 0.00, 1.0]
LEFT_STAND_ARM_POSE = [1.45, 0.10, -1.20, -0.50, 0.00, 1.0]

# Pepper hand values for naoqi_driver2 on this Pepper setup:
#   1.0 = fully open
#   0.0 = fully closed
# The old NAOqi test values were written as “amount closed”, so the ROS value is
# converted as: hand_angle = 1.0 - close_amount.
HAND_OPEN_VALUE = 1.0
HAND_CLOSED_VALUE = 0.0
HAND_REST_VALUE = 0.60
SAFETY_CLOSE_AMOUNT_MAX = 0.40
SQUISH_SMALL_CLOSE_AMOUNT = 0.30
SQUISH_SMALL_SPEED = 0.45
SQUISH_SMALL_HOLD_S = 0.15
SQUISH_LARGE_CLOSE_AMOUNT = 0.42
SQUISH_LARGE_SPEED = 0.60
SQUISH_LARGE_HOLD_S = 0.25
FINISHED_SQUISH_CLOSE_AMOUNT = 0.25
FINISHED_SQUISH_SPEED = 0.40
FINISHED_SQUISH_HOLD_S = 0.40

SAY_INTRO = (
    "Welcome to the Stroop game. Please put your arm on the marked spot, "
    "say your answer out loud, and try not to move."
)
SAY_FINISH = "The task is finished. Please fill in the questionnaire."


def reverse_pose_sequence(seq: Sequence[Tuple[List[float], float, float]]) -> List[Tuple[List[float], float, float]]:
    """Reverse a pose sequence and slow it slightly for safer return motion."""
    return [
        (angles, max(0.05, float(speed) * 0.85), max(0.2, float(hold_s)))
        for angles, speed, hold_s in reversed(list(seq))
    ]


class PepperArmBehavior:
    """Small ROS-only Pepper arm/hand behavior layer for naoqi_driver2."""

    def __init__(self, node: Node):
        self.node = node
        self.side = str(node.get_parameter("pepper_side").value).lower()
        if self.side not in {"left", "right"}:
            node.get_logger().warning("pepper_side must be 'left' or 'right'; falling back to 'right'.")
            self.side = "right"

        self.joint_topic = str(node.get_parameter("pepper_joint_topic").value)
        self.speech_topic = str(node.get_parameter("pepper_speech_topic").value)
        self.auto_place_on_start = bool(node.get_parameter("pepper_auto_place_on_start").value)
        self.speak_enabled = bool(node.get_parameter("pepper_speak_enabled").value)
        self.return_on_finish = bool(node.get_parameter("pepper_return_on_finish").value)
        self.hold_pose_enabled = bool(node.get_parameter("pepper_hold_pose_enabled").value)
        self.hold_pose_period_s = max(0.05, float(node.get_parameter("pepper_hold_pose_period_s").value))
        self.hold_pose_speed = max(0.01, min(1.0, float(node.get_parameter("pepper_hold_pose_speed").value)))
        self.hold_pose_start_delay_s = max(0.0, float(node.get_parameter("pepper_hold_pose_start_delay_s").value))
        self.force_hand_open_during_hold = bool(node.get_parameter("pepper_force_hand_open_during_hold").value)
        self.hand_open_speed = max(0.01, min(1.0, float(node.get_parameter("pepper_hand_open_speed").value)))
        self.return_to_stand_pose = bool(node.get_parameter("pepper_return_to_stand_pose").value)
        self.stand_pose_speed = max(0.01, min(1.0, float(node.get_parameter("pepper_stand_pose_speed").value)))
        self.stand_pose_hold_s = max(0.0, float(node.get_parameter("pepper_stand_pose_hold_s").value))

        self.joint_pub = node.create_publisher(JointAnglesWithSpeed, self.joint_topic, 10)
        self.speech_pub = node.create_publisher(String, self.speech_topic, 10)

        if self.side == "right":
            self.arm_joints = RIGHT_ARM_JOINTS
            self.hand_joint = RIGHT_HAND
            self.place_sequence = RIGHT_TABLE_POSE_SEQUENCE
            self.stand_arm_pose = list(node.get_parameter("pepper_right_stand_arm_pose").value)
        else:
            self.arm_joints = LEFT_ARM_JOINTS
            self.hand_joint = LEFT_HAND
            self.place_sequence = LEFT_TABLE_POSE_SEQUENCE
            self.stand_arm_pose = list(node.get_parameter("pepper_left_stand_arm_pose").value)

        if len(self.stand_arm_pose) != len(self.arm_joints):
            node.get_logger().warning(
                "Stand arm pose must contain 6 values. Falling back to the built-in neutral arm pose."
            )
            self.stand_arm_pose = list(RIGHT_STAND_ARM_POSE if self.side == "right" else LEFT_STAND_ARM_POSE)
        self.stand_arm_pose[-1] = HAND_OPEN_VALUE

        self._busy_until = 0.0
        self._hand_busy_until = 0.0
        self._hold_timer_id = None
        self._hold_joint_names: List[str] = []
        self._hold_angles: List[float] = []

    def busy(self) -> bool:
        return time.monotonic() < self._busy_until

    def _ros_context_is_valid(self) -> bool:
        """Return True only while ROS publishing is still legal."""
        try:
            return rclpy.ok() and self.node.context.ok()
        except Exception:
            return False

    def _safe_log(self, level: str, text: str) -> None:
        """Avoid rosout logging after ROS shutdown has started."""
        try:
            if self._ros_context_is_valid():
                logger = self.node.get_logger()
                if level == "warning":
                    logger.warning(text)
                else:
                    logger.info(text)
            else:
                print("[%s] %s" % (level.upper(), text))
        except Exception:
            print("[%s] %s" % (level.upper(), text))

    def _publish_angles(self, joint_names: Sequence[str], angles: Sequence[float], speed: float, relative: bool = False) -> bool:
        if not self._ros_context_is_valid():
            return False
        msg = JointAnglesWithSpeed()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.joint_names = list(joint_names)
        msg.joint_angles = [float(a) for a in angles]
        msg.speed = float(max(0.01, min(1.0, speed)))
        msg.relative = 1 if relative else 0
        try:
            self.joint_pub.publish(msg)
            return True
        except Exception as exc:
            self._safe_log("warning", "Pepper joint publish skipped/failed: %s" % exc)
            return False

    def say(self, text: str) -> None:
        if not self.speak_enabled or not text or not self._ros_context_is_valid():
            return
        msg = String()
        msg.data = str(text)
        try:
            self.speech_pub.publish(msg)
        except Exception as exc:
            self._safe_log("warning", "Pepper speech publish skipped/failed: %s" % exc)

    def open_hand(self, speed: Optional[float] = None) -> None:
        if speed is None:
            speed = self.hand_open_speed
        self._publish_angles([self.hand_joint], [HAND_OPEN_VALUE], speed)
    
    def rest_hand(self, speed: Optional[float] = None) -> None:
        if speed is None:
            speed = self.hand_open_speed
        self._publish_angles([self.hand_joint], [HAND_REST_VALUE], speed)

    def _close_hand_value(self, close_amount: float, speed: float) -> None:
        close_amount = max(0.0, min(SAFETY_CLOSE_AMOUNT_MAX, float(close_amount)))
        # In this ROS driver setup, 1.0 is open and 0.0 is closed.
        # close_amount keeps the old intuitive tuning scale from the Python 2 test script.
        hand_angle = max(HAND_CLOSED_VALUE, min(HAND_OPEN_VALUE, HAND_OPEN_VALUE - close_amount))
        self._publish_angles([self.hand_joint], [hand_angle], speed)

    def squish(self, close_value: float, speed: float, hold_s: float, reason: str = "") -> None:
        """Close-hold-open using Tk timers so the Stroop GUI does not freeze."""
        if self.busy():
            self.node.get_logger().info("Pepper behavior skipped because another behavior is in progress.")
            return
        self.node.get_logger().info(
            f"Pepper squish {reason}: close={close_value:.2f}, speed={speed:.2f}, hold={hold_s:.2f}s"
        )
        now = time.monotonic()
        self._busy_until = now + max(0.1, float(hold_s) + 0.4)
        self._hand_busy_until = now + max(0.1, float(hold_s) + 0.4)
        self._close_hand_value(close_value, speed)
        # Tk root is owned by the Stroop class; set later via attach_root().
        def reopen() -> None:
            self.open_hand(speed)
            self._hand_busy_until = time.monotonic() + 0.15
        self._root.after(int(max(0.0, hold_s) * 1000.0), reopen)

    def attach_root(self, root: tk.Tk) -> None:
        self._root = root

    def small_squish(self, reason: str = "") -> None:
        self.squish(SQUISH_SMALL_CLOSE_AMOUNT, SQUISH_SMALL_SPEED, SQUISH_SMALL_HOLD_S, reason)

    def large_squish(self, reason: str = "") -> None:
        self.squish(SQUISH_LARGE_CLOSE_AMOUNT, SQUISH_LARGE_SPEED, SQUISH_LARGE_HOLD_S, reason)

    def finished_squish(self) -> None:
        self.squish(FINISHED_SQUISH_CLOSE_AMOUNT, FINISHED_SQUISH_SPEED, FINISHED_SQUISH_HOLD_S, "finished")

    def _cancel_hold_pose(self) -> None:
        if self._hold_timer_id is not None:
            try:
                self._root.after_cancel(self._hold_timer_id)
            except tk.TclError:
                pass
            self._hold_timer_id = None
        self._hold_joint_names = []
        self._hold_angles = []

    def _start_hold_pose(self, full_pose_angles: Sequence[float], label: str) -> None:
        """Continuously re-publish the last arm pose.

        This replaces the old NAOqi test script's pose-hold thread as closely as
        possible using only naoqi_driver2 /joint_angles commands. The hand joint is
        intentionally excluded so squish/open commands can still control the hand.
        """
        if not self.hold_pose_enabled:
            return

        # The pose sequence contains hand as the final joint. Hold only shoulder/elbow/wrist.
        self._hold_joint_names = list(self.arm_joints[:-1])
        self._hold_angles = [float(a) for a in list(full_pose_angles)[: len(self._hold_joint_names)]]

        def tick() -> None:
            if not self._hold_joint_names or not self._hold_angles:
                return
            self._publish_angles(self._hold_joint_names, self._hold_angles, self.hold_pose_speed)
            # Keep the palm fully open while holding the arm pose. Do not do this
            # during an intentional squish, otherwise the hold loop would cancel
            # the close-hold-open feedback action immediately.
            if self.force_hand_open_during_hold and time.monotonic() >= self._hand_busy_until:
                self.open_hand(self.hand_open_speed)
            self._hold_timer_id = self._root.after(
                int(self.hold_pose_period_s * 1000.0),
                tick,
            )

        self.node.get_logger().info(f"Pepper hold-pose started: {label}")
        if self._hold_timer_id is None:
            tick()

    def run_pose_sequence(
        self,
        sequence: Sequence[Tuple[List[float], float, float]],
        label: str,
        hold_final_pose: bool = False,
        cancel_existing_hold: bool = True,
    ) -> None:
        """Run a multi-step arm sequence using Tk timers, non-blocking."""
        if self.busy():
            self.node.get_logger().info(f"Pepper {label} skipped because another behavior is in progress.")
            return

        if cancel_existing_hold:
            self._cancel_hold_pose()

        total_s = sum(max(0.0, float(hold_s)) for _, _, hold_s in sequence) + 0.5
        self._busy_until = time.monotonic() + total_s
        self.node.get_logger().info(f"Pepper starting sequence: {label}")

        def step(i: int) -> None:
            if i >= len(sequence):
                self.node.get_logger().info(f"Pepper sequence complete: {label}")
                # Make sure the hand is open after placement/return. The final
                # pose sequence also contains hand=1.0, but this extra command
                # makes the intended state explicit.
                self.open_hand(self.hand_open_speed)
                if hold_final_pose and sequence:
                    final_angles = list(sequence[-1][0])
                    # Give Pepper time to physically settle at the final pose before
                    # starting the repeated hold commands. Starting the hold loop too
                    # early can make the arm crawl/vibrate toward the target.
                    delay_ms = int(self.hold_pose_start_delay_s * 1000.0)
                    if delay_ms > 0:
                        self._hold_timer_id = self._root.after(
                            delay_ms,
                            lambda: self._start_hold_pose(final_angles, label),
                        )
                    else:
                        self._start_hold_pose(final_angles, label)
                return
            angles, speed, hold_s = sequence[i]
            command_angles = list(angles)
            # For transport sequences, keep Pepper's hand fully open.
            # In this setup, 1.0 = open and 0.0 = closed.
            # This avoids old NAOqi test-pose hand values accidentally closing
            # the hand while the arm is moving to/from the table.
            if len(command_angles) == len(self.arm_joints) and (
                "place arm" in label or "return arm" in label
            ):
                command_angles[-1] = HAND_OPEN_VALUE
            self._publish_angles(self.arm_joints, command_angles, speed)
            self._root.after(int(max(0.0, hold_s) * 1000.0), lambda: step(i + 1))

        step(0)

    def place_arm_on_table(self) -> None:
        self.open_hand()
        self.run_pose_sequence(self.place_sequence, "place arm on table", hold_final_pose=True)

    def return_arm_to_rest(self) -> None:
        self.open_hand()
        self._cancel_hold_pose()
        if self.return_on_finish:
            return_sequence = reverse_pose_sequence(self.place_sequence)
            if self.return_to_stand_pose:
                # First move down/away by reversing the exact table-placement path,
                # then finish in a neutral standing arm pose.
                return_sequence = list(return_sequence) + [
                    (list(self.stand_arm_pose), self.stand_pose_speed, self.stand_pose_hold_s)
                ]
            self.run_pose_sequence(
                return_sequence,
                "return arm to stand pose",
                hold_final_pose=False,
                cancel_existing_hold=False,
            )


    def cleanup_return_to_stand_blocking(self) -> None:
        """Best-effort emergency cleanup for Ctrl+C or closing the GUI.

        The normal return path uses Tk timers, which may not execute after the
        window is destroyed. This method publishes the return sequence
        synchronously before the ROS node/context is destroyed.
        """
        self._safe_log("info", "Pepper emergency cleanup: opening hand and returning to stand pose.")
        try:
            self._cancel_hold_pose()
        except Exception:
            pass

        self._busy_until = 0.0
        self._hand_busy_until = 0.0

        try:
            self.open_hand(self.hand_open_speed)
            time.sleep(0.4)
        except Exception as exc:
            self._safe_log("warning", "Emergency open-hand command failed: %s" % exc)

        if not self.return_on_finish:
            return

        return_sequence = reverse_pose_sequence(self.place_sequence)
        if self.return_to_stand_pose:
            return_sequence = list(return_sequence) + [
                (list(self.stand_arm_pose), self.stand_pose_speed, self.stand_pose_hold_s)
            ]

        for angles, speed, hold_s in return_sequence:
            try:
                command_angles = list(angles)
                if len(command_angles) == len(self.arm_joints):
                    command_angles[-1] = HAND_OPEN_VALUE
                self._publish_angles(self.arm_joints, command_angles, speed)
                time.sleep(max(0.05, float(hold_s)))
            except Exception as exc:
                self._safe_log("warning", "Emergency return step failed: %s" % exc)

        try:
            self.open_hand(self.hand_open_speed)
            time.sleep(0.2)
        except Exception:
            pass


class Stroop(Node):
    def __init__(self):
        super().__init__("stroop_pepper")

        # Existing Stroop parameters.
        self.declare_parameter("ground_truth_mode", "ink")
        self.declare_parameter("total_trials", 30)
        self.declare_parameter("allow_congruent", True)
        self.declare_parameter("colors", DEFAULT_COLORS)
        self.declare_parameter("random_seed", -1)
        self.declare_parameter("gripper_mode", "random")  # random or game
        self.declare_parameter("result_topic", "stroop/result")
        self.declare_parameter("finished_topic", "stroop/finished")
        self.declare_parameter("random_trigger_probability", 0.25)
        self.declare_parameter("random_squish_profile", "weighted")  # small, large, mixed, weighted
        self.declare_parameter("random_large_probability", 1.0 / 3.0)
        self.declare_parameter("mistakes_threshold", 3)
        self.declare_parameter("game_squish_profile", "pattern")  # small, large, mixed, weighted, pattern
        self.declare_parameter("game_squish_pattern", ["small", "small", "large"])

        # Pepper / naoqi_driver2 parameters.
        self.declare_parameter("pepper_side", "right")
        self.declare_parameter("pepper_joint_topic", "/joint_angles")
        self.declare_parameter("pepper_speech_topic", "/speech")
        self.declare_parameter("pepper_auto_place_on_start", True)
        self.declare_parameter("pepper_return_on_finish", True)
        self.declare_parameter("pepper_speak_enabled", False)
        self.declare_parameter("pepper_hold_pose_enabled", True)
        self.declare_parameter("pepper_hold_pose_period_s", 0.50)
        self.declare_parameter("pepper_hold_pose_speed", 0.08)
        self.declare_parameter("pepper_hold_pose_start_delay_s", 1.0)
        self.declare_parameter("pepper_force_hand_open_during_hold", True)
        self.declare_parameter("pepper_hand_open_speed", 0.30)
        self.declare_parameter("pepper_return_to_stand_pose", True)
        self.declare_parameter("pepper_stand_pose_speed", 0.25)
        self.declare_parameter("pepper_stand_pose_hold_s", 1.0)
        self.declare_parameter("pepper_right_stand_arm_pose", RIGHT_STAND_ARM_POSE)
        self.declare_parameter("pepper_left_stand_arm_pose", LEFT_STAND_ARM_POSE)

        self.ground_truth_mode = str(self.get_parameter("ground_truth_mode").value).lower()
        if self.ground_truth_mode not in {"ink", "word"}:
            self.get_logger().warning("ground_truth_mode must be 'ink' or 'word'. Falling back to 'ink'.")
            self.ground_truth_mode = "ink"

        self.total_trials = max(0, int(self.get_parameter("total_trials").value))
        self.allow_congruent = bool(self.get_parameter("allow_congruent").value)
        self.colors = self._normalize_colors(self.get_parameter("colors").value)
        self.gripper_mode = str(self.get_parameter("gripper_mode").value).lower()
        if self.gripper_mode not in {"random", "game"}:
            self.get_logger().warning("gripper_mode must be 'random' or 'game'. Falling back to 'random'.")
            self.gripper_mode = "random"

        self.result_topic = str(self.get_parameter("result_topic").value)
        self.finished_topic = str(self.get_parameter("finished_topic").value)
        self.random_trigger_probability = min(1.0, max(0.0, float(self.get_parameter("random_trigger_probability").value)))
        self.random_squish_profile = str(self.get_parameter("random_squish_profile").value).lower()
        self.random_large_probability = min(1.0, max(0.0, float(self.get_parameter("random_large_probability").value)))
        self.mistakes_threshold = max(1, int(self.get_parameter("mistakes_threshold").value))
        self.game_squish_profile = str(self.get_parameter("game_squish_profile").value).lower()
        self.game_squish_pattern = [
            str(item).lower() for item in list(self.get_parameter("game_squish_pattern").value)
        ]
        self.game_squish_pattern = [
            item for item in self.game_squish_pattern if item in {"small", "large"}
        ] or ["small", "small", "large"]
        self._game_squish_pattern_index = 0

        random_seed = int(self.get_parameter("random_seed").value)
        self._rng = random.Random()
        if random_seed >= 0:
            self._rng.seed(random_seed)

        self.answer_pub = self.create_publisher(Bool, "stroop/answer", 10)
        self.result_pub = self.create_publisher(Bool, self.result_topic, 10)
        self.answer_detail_pub = self.create_publisher(String, "stroop/answer_detail", 10)
        self.trial_pub = self.create_publisher(String, "stroop/trial", 10)
        self.finished_pub = self.create_publisher(Bool, self.finished_topic, 10)

        self._running = False
        self._started = False
        self._trial_id = 0
        self._awaiting_advance_key = True
        self._phase_index = 0
        self._phase_stimuli: List[Dict[str, Any]] = []
        self._phase_stimulus_index = 0
        self._active_timer_id: Optional[str] = None
        self._pending_responses: List[Dict[str, Any]] = []
        self._mistake_count = 0
        self._collect_responses_in_block = True
        self._protocol_phases = self._build_study_phases()
        self._cleanup_started = False

        self._root = tk.Tk()
        self._root.title("Stroop Ground Truth")
        self._root.configure(bg="white")
        self._root.geometry("720x420+880+120")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._mode_label = tk.Label(self._root, text="Stroop Study Protocol", bg="white", fg="black", font=("Helvetica", 18))
        self._mode_label.pack(pady=(20, 8))
        self._truth_label = tk.Label(self._root, text="", bg="white", fg="black", font=("Helvetica", 84, "bold"))
        self._truth_label.pack(expand=True)
        self._controls_label = tk.Label(
            self._root,
            text="Press SPACE to continue\nUse TRUE/FALSE buttons (or J/F) during blocks",
            bg="white",
            fg="black",
            font=("Helvetica", 24),
        )
        self._controls_label.pack(pady=(8, 4))
        self._button_frame = tk.Frame(self._root, bg="white")
        self._button_frame.pack(pady=(2, 6))
        self._true_button = tk.Button(
            self._button_frame,
            text="TRUE (J)",
            font=("Helvetica", 16, "bold"),
            bg="#2e7d32",
            fg="white",
            width=10,
            command=lambda: self._submit_manager_response(True),
        )
        self._true_button.pack(side="left", padx=8)
        self._false_button = tk.Button(
            self._button_frame,
            text="FALSE (F)",
            font=("Helvetica", 16, "bold"),
            bg="#b71c1c",
            fg="white",
            width=10,
            command=lambda: self._submit_manager_response(False),
        )
        self._false_button.pack(side="left", padx=8)
        self._feedback_label = tk.Label(self._root, text="", bg="white", fg="black", font=("Helvetica", 24, "bold"))
        self._feedback_label.pack(pady=(0, 4))
        self._pending_label = tk.Label(self._root, text="Pending: 0", bg="white", fg="#444444", font=("Helvetica", 14))
        self._pending_label.pack(pady=(0, 6))
        self._status_label = tk.Label(self._root, text="Initializing...", bg="white", fg="black", font=("Helvetica", 13))
        self._status_label.pack(pady=(0, 16))
        self._root.bind_all("<KeyPress>", self._on_key_press)

        self._stimulus_window = tk.Toplevel(self._root)
        self._stimulus_window.title("Stroop Stimulus")
        self._stimulus_window.configure(bg="white")
        self._stimulus_window.geometry("720x420+80+120")
        self._stimulus_window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._stimulus_label = tk.Label(self._stimulus_window, text="", bg="white", font=("Helvetica", 100, "bold"))
        self._stimulus_label.pack(expand=True)
        self._instruction_label = tk.Label(
            self._stimulus_window,
            text="Follow the instruction shown on the control screen.",
            bg="white",
            fg="black",
            font=("Helvetica", 20),
        )
        self._instruction_label.pack(pady=(0, 20))

        self.pepper = PepperArmBehavior(self)
        self.pepper.attach_root(self._root)

    def _normalize_colors(self, raw_colors: List[str]) -> List[str]:
        if not isinstance(raw_colors, (list, tuple)):
            raw_colors = [raw_colors]
        normalized = [str(color).strip().lower() for color in raw_colors if str(color).strip()]
        deduplicated = list(dict.fromkeys(normalized))
        if len(deduplicated) < 2:
            self.get_logger().warning("colors must contain at least two names. Falling back to defaults.")
            return DEFAULT_COLORS.copy()
        return deduplicated

    def _publish_trial(self, trial: Dict[str, Any]) -> None:
        msg = String()
        msg.data = ";".join(f"{key}={value}" for key, value in trial.items())
        self.trial_pub.publish(msg)

    def _build_study_phases(self) -> List[Dict[str, Any]]:
        screen_1 = (
            "By participating in this study, you confirm that you are above 18 years old and are not colorblind.\n\n"
            "You will see words of color written in different colors. Your task is to read either the word or the color of the word.\n\n"
            "The robot is close to you.\n\n"
            "At the end of the tasks, you will be asked to fill in a questionnaire. All information you provide will be confidential, and your anonymity will be protected throughout the study."
        )
        screen_2 = "Read the color of the word\n"
        screen_3 = "Read the color or the word depending on the instruction.\nThe blocks will alternate between READ THE WORD and READ THE COLOR."
        return [
            {"type": "instruction", "screen": 1, "title": "Screen 1", "message": screen_1},
            {"type": "instruction", "screen": 2, "title": "Screen 2", "message": screen_2},
            {"type": "block", "screen": 2, "name": "pretest_color_congruent", "task": "read_color", "colors": STUDY_COLORS, "repetitions": 15, "duration_ms": 1000, "congruent": True, "collect_response": False},
            {"type": "instruction", "screen": 3, "title": "Screen 3", "message": screen_3},
            {"type": "break", "screen": 3, "duration_ms": 5000, "message": "READ THE WORD"},
            {"type": "block", "screen": 3, "name": "word_4colors", "task": "read_word", "colors": STUDY_COLORS[:4], "repetitions": 15, "duration_ms": 1200, "congruent": False},
            {"type": "break", "screen": 3, "duration_ms": 5000, "message": "READ THE COLOR"},
            {"type": "block", "screen": 3, "name": "color_4colors", "task": "read_color", "colors": STUDY_COLORS[:4], "repetitions": 15, "duration_ms": 1200, "congruent": False},
            {"type": "break", "screen": 3, "duration_ms": 5000, "message": "READ THE WORD"},
            {"type": "block", "screen": 3, "name": "word_6colors", "task": "read_word", "colors": STUDY_COLORS[:6], "repetitions": 15, "duration_ms": 1000, "congruent": False},
            {"type": "break", "screen": 3, "duration_ms": 5000, "message": "READ THE COLOR"},
            {"type": "block", "screen": 3, "name": "color_6colors", "task": "read_color", "colors": STUDY_COLORS[:6], "repetitions": 15, "duration_ms": 1000, "congruent": False},
            {"type": "break", "screen": 3, "duration_ms": 5000, "message": "READ THE WORD"},
            {"type": "block", "screen": 3, "name": "word_8colors", "task": "read_word", "colors": STUDY_COLORS, "repetitions": 15, "duration_ms": 800, "congruent": False},
            {"type": "break", "screen": 3, "duration_ms": 5000, "message": "READ THE COLOR"},
            {"type": "block", "screen": 3, "name": "color_8colors", "task": "read_color", "colors": STUDY_COLORS, "repetitions": 15, "duration_ms": 800, "congruent": False},
            {"type": "instruction", "screen": 5, "title": "Screen 5", "message": "End. Please fill in the questionnaire", "end_screen": True},
        ]

    def _make_color_sequence(self, colors: List[str], total: int) -> List[str]:
        sequence: List[str] = []
        while len(sequence) < total:
            shuffled = list(colors)
            self._rng.shuffle(shuffled)
            sequence.extend(shuffled)
        return sequence[:total]

    def _build_block_stimuli(self, phase: Dict[str, Any]) -> List[Dict[str, Any]]:
        colors = list(phase["colors"])
        color_sequence = self._make_color_sequence(colors, int(phase["repetitions"]))
        if bool(phase["congruent"]):
            stimuli = [{"word": color, "ink": color} for color in color_sequence]
            self._rng.shuffle(stimuli)
            return stimuli
        if phase["task"] == "read_word":
            return [{"word": word, "ink": self._rng.choice([ink for ink in colors if ink != word])} for word in color_sequence]
        return [{"word": self._rng.choice([word for word in colors if word != ink]), "ink": ink} for ink in color_sequence]

    def _show_instruction_screen(self, title: str, message: str, allow_advance: bool = True) -> None:
        self._awaiting_advance_key = allow_advance
        self._collect_responses_in_block = False
        self._set_manager_controls_enabled(False)
        self._truth_label.config(text=title, fg="black", font=("Helvetica", 40, "bold"))
        self._controls_label.config(text="Press SPACE to continue" if allow_advance else "Use TRUE/FALSE during blocks")
        self._feedback_label.config(text="", fg="black")
        self._update_pending_indicator()
        self._status_label.config(text="Waiting for participant" if allow_advance else "")
        self._set_instruction_layout()
        self._stimulus_window.configure(bg="white")
        self._stimulus_label.config(text="", fg="black", bg="white")
        self._instruction_label.config(text=message, fg="black", bg="white", wraplength=680, justify="left")
        self._root.focus_force()

    def _set_instruction_layout(self) -> None:
        self._stimulus_label.pack_forget()
        self._instruction_label.pack_forget()
        self._instruction_label.pack(expand=True)

    def _set_block_layout(self) -> None:
        self._instruction_label.pack_forget()
        self._stimulus_label.pack_forget()
        self._stimulus_label.pack(expand=True)
        self._instruction_label.pack(pady=(0, 20))

    def _publish_current_stimulus(self, phase: Dict[str, Any], stimulus: Dict[str, str]) -> None:
        self._trial_id += 1
        ground_truth = stimulus["word"] if phase["task"] == "read_word" else stimulus["ink"]
        trial = {
            "id": self._trial_id,
            "screen": phase["screen"],
            "phase": phase["name"],
            "task": phase["task"],
            "duration_ms": phase["duration_ms"],
            "word": stimulus["word"],
            "ink": stimulus["ink"],
            "ground_truth": ground_truth,
        }
        self._publish_trial(trial)
        if not self._collect_responses_in_block:
            self._truth_label.config(text="CONTROL", fg="black", font=("Helvetica", 54, "bold"))
            return
        self._truth_label.config(text=ground_truth.upper(), fg="black", font=("Helvetica", 84, "bold"))
        duration_ms = int(phase["duration_ms"])
        grace_ms = max(1, int(duration_ms * 0.5))
        pending = {"trial": trial, "stimulus": stimulus, "answered": False, "start_ts": time.monotonic(), "deadline_timer_id": None}
        pending["deadline_timer_id"] = self._root.after(duration_ms + grace_ms, lambda trial_id=trial["id"]: self._auto_false_if_pending(trial_id))
        self._pending_responses.append(pending)
        self._update_pending_indicator()

    def _trim_answered_pending(self) -> None:
        self._pending_responses = [p for p in self._pending_responses if not p.get("answered", False)]
        self._update_pending_indicator()

    def _update_pending_indicator(self) -> None:
        pending_count = sum(1 for pending in self._pending_responses if not pending.get("answered", False))
        self._pending_label.config(text=f"Pending: {pending_count}")

    def _flash_feedback(self, value: bool) -> None:
        self._feedback_label.config(text="TRUE" if value else "FALSE", fg="#2e7d32" if value else "#b71c1c")
        self._root.after(700, lambda: self._feedback_label.config(text="", fg="black"))

    def _trigger_pepper_feedback(self, profile: str, reason: str) -> None:
        profile = profile.lower()
        if profile == "large":
            self.pepper.large_squish(reason)
        elif profile == "mixed":
            # Optional 50/50 mode.
            if self._rng.random() < 0.5:
                self.pepper.small_squish(reason)
            else:
                self.pepper.large_squish(reason)
        elif profile == "weighted":
            # User-requested random mode: 1/3 large, 2/3 small by default.
            if self._rng.random() < self.random_large_probability:
                self.pepper.large_squish(reason)
            else:
                self.pepper.small_squish(reason)
        else:
            self.pepper.small_squish(reason)

    def _trigger_game_pattern_feedback(self, reason: str) -> None:
        # User-requested game mode: small, small, large, repeating.
        profile = self.game_squish_pattern[
            self._game_squish_pattern_index % len(self.game_squish_pattern)
        ]
        self._game_squish_pattern_index += 1
        self._trigger_pepper_feedback(profile, f"{reason} ({profile})")

    def _handle_result_feedback(self, answer: bool) -> None:
        if self.gripper_mode == "random":
            if self._rng.random() <= self.random_trigger_probability:
                self._trigger_pepper_feedback(self.random_squish_profile, "random mode trigger")
            return
        if answer:
            return
        self._mistake_count += 1
        if self._mistake_count >= self.mistakes_threshold:
            self._mistake_count = 0
            if self.game_squish_profile == "pattern":
                self._trigger_game_pattern_feedback("game threshold reached")
            else:
                self._trigger_pepper_feedback(self.game_squish_profile, "game threshold reached")

    def _record_response_for_pending(self, pending: Dict[str, Any], answer: bool, source: str) -> None:
        if pending.get("answered", False):
            return
        pending["answered"] = True
        timer_id = pending.get("deadline_timer_id")
        if timer_id is not None:
            try:
                self._root.after_cancel(timer_id)
            except tk.TclError:
                pass
            pending["deadline_timer_id"] = None
        elapsed_ms = int((time.monotonic() - pending["start_ts"]) * 1000)
        trial = pending["trial"]
        answer_msg = Bool()
        answer_msg.data = answer
        self.answer_pub.publish(answer_msg)
        result_msg = Bool()
        result_msg.data = answer
        self.result_pub.publish(result_msg)
        detail_msg = String()
        detail_msg.data = (
            f"id={trial['id']};answer={'true' if answer else 'false'};"
            f"source={source};response_time_ms={elapsed_ms};"
            f"word={trial['word']};ink={trial['ink']};"
            f"ground_truth={trial['ground_truth']};phase={trial['phase']};screen={trial['screen']};task={trial['task']}"
        )
        self.answer_detail_pub.publish(detail_msg)
        self._status_label.config(text=f"Trial {trial['id']} recorded: {'TRUE' if answer else 'FALSE'} ({source}, {elapsed_ms} ms)")
        self._flash_feedback(answer)
        self._handle_result_feedback(answer)
        self._trim_answered_pending()

    def _record_response(self, answer: bool, source: str) -> None:
        for pending in self._pending_responses:
            if not pending.get("answered", False):
                self._record_response_for_pending(pending, answer, source)
                return

    def _auto_false_if_pending(self, trial_id: int) -> None:
        for pending in self._pending_responses:
            if pending["trial"]["id"] == trial_id and not pending.get("answered", False):
                self._record_response_for_pending(pending, False, "timeout")
                return

    def _submit_manager_response(self, answer: bool) -> None:
        if self._pending_responses:
            self._record_response(answer, "manager")

    def _start_protocol(self) -> None:
        self._started = True
        self._phase_index = 0
        self._set_manager_controls_enabled(False)
        self._controls_label.config(text="")
        self._status_label.config(text="Starting protocol: sending Pepper table-placement command...")
        self.pepper.say(SAY_INTRO)
        if self.pepper.auto_place_on_start:
            self.get_logger().info(
                "SPACE pressed: sending Pepper arm-to-table sequence. "
                "If the arm does not move and feels limp, run ALMotion.wakeUp "
                "and setStiffnesses before starting this script."
            )
            self.pepper.place_arm_on_table()
        self._advance_phase()

    def _advance_phase(self) -> None:
        if not self._running:
            return
        if self._active_timer_id is not None:
            try:
                self._root.after_cancel(self._active_timer_id)
            except tk.TclError:
                pass
            self._active_timer_id = None
        for pending in self._pending_responses:
            timer_id = pending.get("deadline_timer_id")
            if timer_id is not None:
                try:
                    self._root.after_cancel(timer_id)
                except tk.TclError:
                    pass
        self._pending_responses = []
        self._update_pending_indicator()
        if self._phase_index >= len(self._protocol_phases):
            self._finish_protocol()
            return
        phase = self._protocol_phases[self._phase_index]
        self._phase_index += 1
        if phase["type"] == "instruction":
            allow_advance = not bool(phase.get("end_screen", False))
            self._show_instruction_screen(phase["title"], phase["message"], allow_advance=allow_advance)
            if phase.get("end_screen", False):
                self._finish_protocol()
            return
        if phase["type"] == "break":
            self._awaiting_advance_key = False
            self._collect_responses_in_block = False
            self._set_manager_controls_enabled(False)
            cue_text = str(phase.get("message", "BREAK"))
            self._truth_label.config(text=cue_text, fg="black", font=("Helvetica", 60, "bold"))
            self._controls_label.config(text="")
            self._feedback_label.config(text="", fg="black")
            self._status_label.config(text=f"Black screen for {phase['duration_ms'] // 1000} seconds")
            self._stimulus_window.configure(bg="black")
            self._stimulus_label.config(text="", fg="white", bg="black")
            self._instruction_label.config(text=cue_text, fg="white", bg="black", font=("Helvetica", 24, "bold"))
            self._active_timer_id = self._root.after(int(phase["duration_ms"]), self._advance_phase)
            return
        if phase["type"] == "block":
            self._awaiting_advance_key = False
            self._collect_responses_in_block = bool(phase.get("collect_response", True))
            self._phase_stimuli = self._build_block_stimuli(phase)
            self._phase_stimulus_index = 0
            instruction = "Read the WORD" if phase["task"] == "read_word" else "Read the COLOR"
            self._truth_label.config(text=instruction, fg="black", font=("Helvetica", 54, "bold"))
            self._set_manager_controls_enabled(self._collect_responses_in_block)
            self._controls_label.config(text="Use TRUE/FALSE buttons (or J/F)" if self._collect_responses_in_block else "Control block: no TRUE/FALSE input")
            self._status_label.config(text=f"Screen {phase['screen']} | {phase['name']} | {len(self._phase_stimuli)} stimuli at {phase['duration_ms']} ms")
            self._set_block_layout()
            self._stimulus_window.configure(bg="white")
            self._instruction_label.config(text="Say the response out loud.", fg="black", bg="white", wraplength=680)
            self._show_next_block_stimulus(phase)
            return

    def _show_next_block_stimulus(self, phase: Dict[str, Any]) -> None:
        if not self._running:
            return
        if self._phase_stimulus_index >= len(self._phase_stimuli):
            self._advance_phase()
            return
        stimulus = self._phase_stimuli[self._phase_stimulus_index]
        self._phase_stimulus_index += 1
        self._stimulus_label.config(text=stimulus["word"].upper(), fg=stimulus["ink"], bg="white")
        self._publish_current_stimulus(phase, stimulus)
        self._active_timer_id = self._root.after(int(phase["duration_ms"]), lambda: self._show_next_block_stimulus(phase))

    def _set_manager_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._true_button.config(state=state)
        self._false_button.config(state=state)

    def _finish_protocol(self) -> None:
        self._awaiting_advance_key = False
        self._set_manager_controls_enabled(False)
        self._controls_label.config(text="")
        self._feedback_label.config(text="", fg="black")
        self._status_label.config(text="Protocol complete. Close either window to exit.")
        self.pepper.say(SAY_FINISH)
        self.pepper.finished_squish()
        # Delay return slightly so the finish squish can open first.
        self._root.after(1200, self.pepper.return_arm_to_rest)
        finished_msg = Bool()
        finished_msg.data = True
        self.finished_pub.publish(finished_msg)

    def _on_key_press(self, event) -> str:
        if not self._running:
            return "break"
        key = event.keysym.lower()
        if not self._started:
            if key in {"space", "return", "kp_enter"}:
                self._start_protocol()
            return "break"
        if self._awaiting_advance_key and key in {"space", "return", "kp_enter"}:
            self._controls_label.config(text="")
            self._status_label.config(text="Continuing...")
            self._advance_phase()
            return "break"
        if key == "j":
            self._submit_manager_response(True)
            return "break"
        if key == "f":
            self._submit_manager_response(False)
            return "break"
        return "break"

    def _spin_ros(self) -> None:
        if not self._running:
            return
        rclpy.spin_once(self, timeout_sec=0.0)
        try:
            self._root.after(20, self._spin_ros)
        except tk.TclError:
            self._running = False

    def _cleanup_pepper_blocking(self) -> None:
        """Run Pepper return-to-stand once, before ROS shutdown."""
        if getattr(self, "_cleanup_started", False):
            return
        self._cleanup_started = True
        try:
            self.pepper.cleanup_return_to_stand_blocking()
        except Exception as exc:
            try:
                if rclpy.ok() and self.context.ok():
                    self.get_logger().warning(f"Cleanup Pepper return-to-stand failed: {exc}")
                else:
                    print(f"Cleanup Pepper return-to-stand failed: {exc}")
            except Exception:
                print(f"Cleanup Pepper return-to-stand failed: {exc}")

    def _on_close(self) -> None:
        self._running = False
        if self._active_timer_id is not None:
            try:
                self._root.after_cancel(self._active_timer_id)
            except tk.TclError:
                pass
            self._active_timer_id = None
        for pending in self._pending_responses:
            timer_id = pending.get("deadline_timer_id")
            if timer_id is not None:
                try:
                    self._root.after_cancel(timer_id)
                except tk.TclError:
                    pass
        self._pending_responses = []
        self._cleanup_pepper_blocking()
        try:
            self._root.quit()
            self._root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        self._running = True
        self.pepper.open_hand()
        self._set_manager_controls_enabled(False)
        self._show_instruction_screen(
            "Stroop Study",
            "Press SPACE to begin. The protocol will start with Screen 1 instructions.",
            allow_advance=True,
        )
        self._status_label.config(text="Press SPACE to begin the protocol")
        self._controls_label.config(text="Press SPACE to continue\nUse TRUE/FALSE buttons (or J/F) during blocks")
        self._root.after(20, self._spin_ros)
        self._root.mainloop()


def main(args=None):
    # Disable rclpy's default SIGINT handler when possible. Otherwise Ctrl+C can
    # shut down the ROS context before Pepper cleanup publishes its return commands.
    if SignalHandlerOptions is not None:
        try:
            rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
        except TypeError:
            rclpy.init(args=args)
    else:
        rclpy.init(args=args)

    node = Stroop()
    try:
        node.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: returning Pepper to stand pose before shutdown...")
    finally:
        try:
            node._running = False
            # Must run before destroy_node() and before rclpy.shutdown().
            node._cleanup_pepper_blocking()
        except Exception as exc:
            print(f"Final cleanup failed: {exc}")
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
