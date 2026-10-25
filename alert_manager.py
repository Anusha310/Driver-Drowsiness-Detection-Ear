# alert_manager.py
# Multi-tier alert escalation system with smart cooldown

import time
import pygame
import os

# Alert tiers
TIER_NONE = 0
TIER_VISUAL = 1      # Amber visual warning only
TIER_AUDIO_SOFT = 2   # Short beep + visual
TIER_FULL_ALARM = 3   # Continuous alarm + emergency visual

# State mapping
STATE_AWAKE = 0
STATE_WARNING = 1
STATE_ALERT = 2


class AlertManager:
    """
    Intelligent alert system with tiered escalation and cooldown.

    Tier 1: Visual warning (score 0.28-0.40) — amber UI indicators
    Tier 2: Soft audio + visual (score 0.40-0.55) — beep pattern
    Tier 3: Full alarm (score > 0.55) — continuous alarm + red overlay

    After alarm stops, a 10-second heightened-sensitivity cooldown prevents
    the driver from repeatedly nodding off and waking just before alarm.
    """

    def __init__(self, alarm_path="alert.mp3"):
        self.alarm_path = alarm_path
        self.current_tier = TIER_NONE
        self.alarm_active = False

        # Cooldown
        self._cooldown_until = 0.0
        self._cooldown_duration = 10.0  # seconds
        self._alert_count = 0

        # Sound init
        self._sound_initialized = False
        self._init_sound()

        # Timing for beep patterns
        self._last_beep_time = 0.0
        self._beep_interval = 1.5  # seconds between soft beeps

    def _init_sound(self):
        """Initialize pygame mixer and pre-load alarm."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            if os.path.exists(self.alarm_path):
                pygame.mixer.music.load(self.alarm_path)
                self._sound_initialized = True
        except Exception as e:
            print(f"Alert sound init failed: {e}")
            self._sound_initialized = False

    def update(self, state, drowsiness_score):
        """
        Update alert tier based on current detection state and score.

        Args:
            state: STATE_AWAKE, STATE_WARNING, or STATE_ALERT
            drowsiness_score: float 0.0 - 1.0
        """
        now = time.time()
        in_cooldown = now < self._cooldown_until

        # Determine target tier
        if state == STATE_ALERT:
            target_tier = TIER_FULL_ALARM
        elif state == STATE_WARNING:
            # During cooldown, escalate warnings faster
            if in_cooldown and drowsiness_score > 0.35:
                target_tier = TIER_FULL_ALARM
            elif drowsiness_score > 0.65: # Raised from 0.40
                target_tier = TIER_AUDIO_SOFT
            else:
                target_tier = TIER_VISUAL
        else:
            target_tier = TIER_NONE

        # Apply tier changes
        if target_tier != self.current_tier:
            self._transition_to(target_tier, now)

        # Handle ongoing tier behaviors
        if self.current_tier == TIER_AUDIO_SOFT:
            self._handle_soft_beep(now)

    def _transition_to(self, new_tier, now):
        """Handle tier transitions."""
        old_tier = self.current_tier

        if new_tier == TIER_FULL_ALARM:
            self._start_alarm()
        elif new_tier == TIER_AUDIO_SOFT:
            self._stop_alarm()
            self._last_beep_time = 0  # trigger immediate beep
        elif new_tier == TIER_VISUAL:
            self._stop_alarm()
        elif new_tier == TIER_NONE:
            if old_tier >= TIER_FULL_ALARM:
                # Entering cooldown after full alarm
                self._cooldown_until = now + self._cooldown_duration
                self._alert_count += 1
            self._stop_alarm()

        self.current_tier = new_tier

    def _start_alarm(self):
        """Start continuous alarm sound."""
        if not self.alarm_active and self._sound_initialized:
            try:
                pygame.mixer.music.load(self.alarm_path)
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play(-1)  # loop
                self.alarm_active = True
            except Exception:
                pass

    def _stop_alarm(self):
        """Stop alarm sound."""
        if self.alarm_active:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            self.alarm_active = False

    def _handle_soft_beep(self, now):
        """Play periodic short beeps for Tier 2."""
        if not self._sound_initialized:
            return
        if now - self._last_beep_time >= self._beep_interval:
            try:
                pygame.mixer.music.load(self.alarm_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(0)  # play once
                self._last_beep_time = now
            except Exception:
                pass

    def force_stop(self):
        """Manually stop all alerts (user pressed S)."""
        self._stop_alarm()
        self.current_tier = TIER_NONE

    def is_in_cooldown(self):
        """Check if currently in post-alert cooldown."""
        return time.time() < self._cooldown_until

    @property
    def alert_count(self):
        return self._alert_count

    def cleanup(self):
        """Cleanup on exit."""
        self._stop_alarm()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
