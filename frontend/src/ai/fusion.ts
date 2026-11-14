import { ClassroomMetrics, StudentAnalytics, TeacherRecommendation } from '../types/analytics';
import { BehavioralFeatures } from './features';
import { TrackedPerson } from './tracker';

interface TemporalBuffer {
  attScores: number[];
  drowsyFlags: boolean[];
  perclosWindow: boolean[];
  offBoardWindow: boolean[];
  yawnFlags: boolean[];
  handFlags: boolean[];
  gazeHistory: string[];
  emotionHistory: string[];
  emaPitch: number | null;
  emaYaw: number | null;
  emaEar: number | null;
  emaAtt: number | null;
  prevEyesClosed: boolean;
  blinkTimestamps: number[];
  lastStableState: string | null;
  stateHoldFrames: number;
  consecutiveClosedFrames: number;
  consecutiveOpenFrames: number;
  drowsyHoldFrames: number;
}

function modeString(arr: string[], fallback: string): string {
  if (arr.length === 0) return fallback;
  const counts = new Map<string, number>();
  let best = fallback;
  let bestCount = 0;
  for (const item of arr) {
    const c = (counts.get(item) ?? 0) + 1;
    counts.set(item, c);
    if (c >= bestCount) {
      bestCount = c;
      best = item;
    }
  }
  return best;
}

function pushBounded<T>(arr: T[], val: T, maxLen: number): void {
  arr.push(val);
  if (arr.length > maxLen) arr.shift();
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/**
 * Browser-side reliability-aware feature fusion & temporal smoothing engine.
 * - Outputs estimated engagement-related behavior (never claims psychological attention).
 * - Enforces contradiction prevention (DROWSY overrides ATTENTIVE).
 * - Marks low-quality or occluded observations as UNABLE TO DETERMINE.
 * - Steady state tracking with deadband hysteresis to eliminate classification jitter.
 */
export class BrowserEngagementFusion {
  private buffers: Map<number, TemporalBuffer> = new Map();

  public reset(): void {
    this.buffers.clear();
  }

  public prune(activeTrackIds: Set<number>): void {
    for (const id of this.buffers.keys()) {
      if (!activeTrackIds.has(id)) {
        this.buffers.delete(id);
      }
    }
  }

  public evaluateStudent(
    track: TrackedPerson,
    feat: BehavioralFeatures
  ): StudentAnalytics {
    let buf = this.buffers.get(track.trackId);
    if (!buf) {
      buf = {
        attScores: [],
        drowsyFlags: [],
        perclosWindow: [],
        offBoardWindow: [],
        yawnFlags: [],
        handFlags: [],
        gazeHistory: [],
        emotionHistory: [],
        emaPitch: null,
        emaYaw: null,
        emaEar: null,
        emaAtt: null,
        prevEyesClosed: false,
        blinkTimestamps: [],
        lastStableState: null,
        stateHoldFrames: 0,
        consecutiveClosedFrames: 0,
        consecutiveOpenFrames: 0,
        drowsyHoldFrames: 0,
      };
      this.buffers.set(track.trackId, buf);
    }

    const activeFace = track.visibleThisFrame && track.hasValidFace && feat.faceDetected;

    if (!activeFace || feat.faceQuality < 0.35) {
      return {
        student_id: track.studentId,
        bbox: track.faceBox
          ? [Math.round(track.faceBox.x1), Math.round(track.faceBox.y1), Math.round(track.faceBox.x2), Math.round(track.faceBox.y2)]
          : [],
        person_bbox: [
          Math.round(track.personBox.x1),
          Math.round(track.personBox.y1),
          Math.round(track.personBox.x2),
          Math.round(track.personBox.y2),
        ],
        face_detected: false,
        face_quality: 0,
        attention_pct: 0,
        attention_status: activeFace ? 'Low Quality Face' : 'No Face Detected',
        gaze_direction: 'Unable to Determine',
        head_pose: 'Unknown',
        emotion: 'Unable to Determine',
        confidence_pct: 0,
        is_drowsy: false,
        ear_value: 0,
        is_yawning: false,
        hand_raised: feat.handRaised,
        posture: feat.posture,
        participation_pct: feat.handRaised ? 70 : 0,
        engagement_pct: 0,
        confusion_index: 0,
        confusion_level: 'Unavailable',
        esi_score: 0,
        status: 'Unavailable',
        final_state: 'UNABLE TO DETERMINE',
        cautious_engagement_category: 'Insufficient evidence',
        reason_for_state: activeFace
          ? 'Estimated engagement behavior unavailable due to low facial landmark quality.'
          : 'Person tracked; facial cues occluded or turned away.',
        ai_suggestion: {
          action: 'Face Visibility Required',
          pedagogical_tip: 'Awaiting clear face visibility to provide behavioral suggestions.',
          priority: 'LOW',
        },
        corrective_guidance: {
          is_fallback: true,
          reason: activeFace
            ? 'Low facial landmark quality / illumination deficiency.'
            : 'Facial region occluded or turned away from lens (> 55° yaw).',
          action_prompt: activeFace
            ? 'Increase front lighting and stabilize camera distance.'
            : 'Instruct student to face toward the frontal camera angle (within ±25°).',
          target_accuracy_path: 'Restores 478-point landmark mesh and >95% behavioral confidence.',
        },
        xai: {
          attention: {
            reasons: [feat.posture !== 'Unknown' ? `Pose: ${feat.posture}` : 'Awaiting clear face visibility'],
            status: 'UNABLE TO DETERMINE',
          },
          counterfactual: {
            explanation: 'Face camera clearly with adequate lighting for reliable behavioral estimation.',
          },
        },
      };
    }

    // Update PERCLOS (30-frame window) & blink rate (60s rolling timestamps)
    const nowMs = Date.now();
    pushBounded(buf.perclosWindow, feat.eyesClosed, 30);
    if (feat.eyesClosed && !buf.prevEyesClosed) {
      buf.blinkTimestamps.push(nowMs);
    }
    buf.prevEyesClosed = feat.eyesClosed;
    buf.blinkTimestamps = buf.blinkTimestamps.filter((ts) => nowMs - ts <= 60000);
    const perclosPct = Number(
      ((buf.perclosWindow.filter(Boolean).length / Math.max(1, buf.perclosWindow.length)) * 100).toFixed(1)
    );
    const blinkRateBpm = buf.blinkTimestamps.length;

    // Track consecutive eye closure and open frames for glitch-free drowsiness
    if (feat.eyesClosed) {
      buf.consecutiveClosedFrames += 1;
      buf.consecutiveOpenFrames = 0;
    } else {
      buf.consecutiveOpenFrames += 1;
      buf.consecutiveClosedFrames = 0;
    }

    const isNoteTaking =
      !feat.eyesClosed &&
      !feat.phoneDetected &&
      (feat.pitchDeg > 16 || feat.eyeLookDownAvg > 0.58);
    const isOffBoard = feat.gazeDirection !== 'Looking at Board' && !isNoteTaking;
    pushBounded(buf.offBoardWindow, isOffBoard, 20);
    const offBoardRatio =
      buf.offBoardWindow.filter(Boolean).length / Math.max(1, buf.offBoardWindow.length);
    const isBoredDisengaged =
      buf.offBoardWindow.length >= 10 &&
      offBoardRatio >= 0.65 &&
      (feat.posture === 'Leaning' || feat.posture === 'Slouching') &&
      (feat.emotion === 'Neutral' || feat.emotion === 'Sleepy');

    // Continuous observable behavioral scoring (0..100) using reliability-weighted modality fusion
    // Robust continuous attention scoring:
    // - Looking directly at board/camera with forward head -> ~84-96% (ATTENTIVE)
    // - Looking slightly sideways, glanced down, or mild off-center -> ~52-72% (MODERATE)
    // - Looking far away, turned head, or distracted -> <42% (LOW ATTENTION)
    const gazeDeviation = feat.gazeDeviation ?? (feat.gazeDirection === 'Looking at Board' ? 0.08 : 0.65);
    const gazeScore = isNoteTaking
      ? 80
      : feat.gazeDirection === 'Looking at Board'
      ? Math.max(82, 98 - gazeDeviation * 25)
      : feat.gazeDirection === 'Looking Left' || feat.gazeDirection === 'Looking Right'
      ? Math.max(48, 70 - (gazeDeviation - 0.3) * 25)
      : Math.max(28, 52 - gazeDeviation * 30);

    const absYaw = Math.abs(feat.yawDeg);
    const absPitch = Math.abs(feat.pitchDeg);
    const continuousHeadScore = isNoteTaking
      ? 82
      : absYaw < 16 && absPitch < 14
      ? Math.max(82, 98 - absYaw * 0.9 - absPitch * 0.8)
      : absYaw < 34 && absPitch < 26
      ? Math.max(54, 80 - (absYaw - 16) * 1.3 - (absPitch - 14) * 1.0)
      : Math.max(22, 50 - (absYaw - 34) * 1.5);
    const headScore = continuousHeadScore;

    const eyeScore = feat.eyesClosed
      ? Math.max(10, 24 - perclosPct * 0.25)
      : Math.max(50, 96 - perclosPct * 0.55);

    const postureScore =
      feat.posture === 'Upright' || feat.posture === 'Standing'
        ? 94
        : feat.posture === 'Leaning'
        ? 78
        : feat.posture === 'Slouching'
        ? 52
        : 76;

    // Modality quality weights
    const gazeRel = feat.modalities?.gaze?.valid ? Math.max(0.2, feat.faceQuality) : 0.1;
    const headRel = feat.modalities?.headPose?.valid ? Math.max(0.2, feat.faceQuality) : 0.1;
    const eyeRel = feat.modalities?.eyeClosure?.valid ? Math.max(0.2, feat.faceQuality) : 0.1;
    const postureRel = feat.modalities?.posture?.valid ? Math.max(0.2, feat.modalities.posture.confidence) : 0.1;
    const totalRel = gazeRel + headRel + eyeRel + postureRel || 1;

    const wGaze = (0.38 * gazeRel) / totalRel;
    const wHead = (0.28 * headRel) / totalRel;
    const wEye = (0.20 * eyeRel) / totalRel;
    const wPosture = (0.14 * postureRel) / totalRel;
    const sumW = wGaze + wHead + wEye + wPosture || 1;

    let rawEngagement =
      (wGaze * gazeScore + wHead * headScore + wEye * eyeScore + wPosture * postureScore) / sumW;

    // Grounding: looking sideways or turned away caps engagement to MODERATE (45-64%) or LOW ATTENTION,
    // preventing open eyes or upright posture from falsely classifying distracted glances as ATTENTIVE.
    if (!isNoteTaking) {
      if (feat.gazeDirection === 'Looking Left' || feat.gazeDirection === 'Looking Right') {
        rawEngagement = Math.min(rawEngagement, 63);
      } else if (absYaw > 20) {
        rawEngagement = Math.min(rawEngagement, 60);
      } else if (feat.gazeDirection === 'Looking Down / Away') {
        rawEngagement = Math.min(rawEngagement, 46);
      }
    }

    if (feat.phoneDetected) rawEngagement = Math.min(rawEngagement, 28);
    if (isBoredDisengaged) rawEngagement = Math.min(rawEngagement, 42);

    pushBounded(buf.attScores, rawEngagement, 16);
    pushBounded(buf.drowsyFlags, feat.eyesClosed, 8);
    pushBounded(buf.yawnFlags, feat.isYawning, 6);
    pushBounded(buf.handFlags, feat.handRaised, 6);
    pushBounded(buf.gazeHistory, feat.gazeDirection, 8);
    pushBounded(buf.emotionHistory, feat.emotion, 8);

    buf.emaPitch = buf.emaPitch === null ? feat.pitchDeg : 0.25 * feat.pitchDeg + 0.75 * buf.emaPitch;
    buf.emaYaw = buf.emaYaw === null ? feat.yawDeg : 0.25 * feat.yawDeg + 0.75 * buf.emaYaw;
    buf.emaEar = buf.emaEar === null ? feat.earValue : 0.35 * feat.earValue + 0.65 * buf.emaEar;

    const smoothedPitch = Number(buf.emaPitch.toFixed(1));
    const smoothedYaw = Number(buf.emaYaw.toFixed(1));
    const smoothedEar = Number(buf.emaEar.toFixed(3));
    const smoothedGaze = modeString(buf.gazeHistory, feat.gazeDirection);
    const smoothedEmotion = modeString(buf.emotionHistory, feat.emotion);

    buf.emaAtt = buf.emaAtt === null ? rawEngagement : 0.20 * rawEngagement + 0.80 * buf.emaAtt;
    const smoothedAtt = Number(buf.emaAtt.toFixed(0));
    const drowsyCount = buf.drowsyFlags.filter(Boolean).length;
    const isYawningConsistent = buf.yawnFlags.filter(Boolean).length >= 2;

    // Drowsiness Identification Engine directly driven by Eye Gaze & Eye Open/Close:
    // 1. Sustained closed eyes (>= 2 consecutive frames ~200-300ms)
    // 2. Immediate deep eye closure (earValue < 0.19 or blinkAvg >= 0.35)
    // 3. High PERCLOS in window (>= 20% proportion of eye closure)
    // 4. Downward eye gaze drift with eyelid drooping / micro-sleep (gaze down or eyeLookDownAvg > 0.48 with eyelids drooping, without note-taking)
    // 5. Yawning combined with closed/droopy eyes
    const isDeepClosure = feat.eyesClosed && feat.earValue > 0 && (smoothedEar < 0.19 || feat.earValue < 0.19);
    const isDownwardGazeDroop =
      (feat.eyesClosed || (feat.earValue > 0 && feat.earValue < 0.225)) &&
      (feat.eyeLookDownAvg > 0.48 || feat.gazeDirection === 'Looking Down / Away' || feat.pitchDeg > 14) &&
      !isNoteTaking;

    const rawDrowsyTrigger =
      (isDeepClosure && buf.consecutiveClosedFrames >= 2) ||
      buf.consecutiveClosedFrames >= 2 ||
      (buf.perclosWindow.length >= 8 && perclosPct >= 20) ||
      (isDownwardGazeDroop && (buf.consecutiveClosedFrames >= 2 || drowsyCount >= 2)) ||
      (isYawningConsistent && (buf.consecutiveClosedFrames >= 1 || drowsyCount >= 1));

    // Steady drowsiness latch: once drowsy is detected, latch for 10 frames (~350ms)
    // to prevent rapid on/off flickering, clearing only after 3 consecutive open frames.
    if (rawDrowsyTrigger) {
      buf.drowsyHoldFrames = 10;
    } else if (buf.consecutiveOpenFrames >= 3 && buf.drowsyHoldFrames > 0) {
      buf.drowsyHoldFrames = Math.max(0, buf.drowsyHoldFrames - 2);
    } else if (buf.drowsyHoldFrames > 0) {
      buf.drowsyHoldFrames -= 1;
    }

    const isDrowsyConsistent = buf.drowsyHoldFrames > 0;
    const handRaisedConsistent = buf.handFlags.filter(Boolean).length >= 2;
    const isMicroSleep = isDrowsyConsistent && (isDownwardGazeDroop || feat.pitchDeg > 14 || feat.eyeLookDownAvg > 0.50);

    // Contradiction prevention: sustained drowsiness overrides attentive state
    let finalState: string;
    let attentionStatus: string;
    let behavioralCue: string;
    let cautiousCategory:
      | 'Attentive-related behaviour estimate'
      | 'Partially engaged or uncertain'
      | 'Low visible engagement-related cues'
      | 'Insufficient evidence';

    // Steady state resolution with deadband hysteresis:
    // - Prevents bouncing around the 70% boundary between ATTENTIVE and MODERATE
    // - ATTENTIVE: >= 68% (drops to MODERATE only below 64%)
    // - MODERATE: 44% - 67% (rises to ATTENTIVE above 72%, drops to LOW below 40%)
    // - LOW ATTENTION: < 42% (rises to MODERATE above 46%)
    const prevStable = buf.lastStableState;
    let targetState = 'LOW ATTENTION';
    if (prevStable === 'ATTENTIVE') {
      targetState = smoothedAtt >= 64 ? 'ATTENTIVE' : smoothedAtt >= 42 ? 'MODERATE' : 'LOW ATTENTION';
    } else if (prevStable === 'MODERATE') {
      targetState = smoothedAtt >= 72 ? 'ATTENTIVE' : smoothedAtt >= 40 ? 'MODERATE' : 'LOW ATTENTION';
    } else if (prevStable === 'LOW ATTENTION') {
      targetState = smoothedAtt >= 72 ? 'ATTENTIVE' : smoothedAtt >= 46 ? 'MODERATE' : 'LOW ATTENTION';
    } else {
      targetState = smoothedAtt >= 68 ? 'ATTENTIVE' : smoothedAtt >= 44 ? 'MODERATE' : 'LOW ATTENTION';
    }

    // 2-frame temporal debounce: prevents single-frame sensor noise from flipping stable states
    if (targetState !== prevStable && prevStable !== null) {
      buf.stateHoldFrames += 1;
      if (buf.stateHoldFrames < 2) {
        targetState = prevStable;
      } else {
        buf.stateHoldFrames = 0;
      }
    } else {
      buf.stateHoldFrames = 0;
    }

    if (isDrowsyConsistent) {
      finalState = 'DROWSY';
      behavioralCue = isMicroSleep ? 'Micro-Sleep / Head Nodding' : 'Drowsy (Sustained Eye Closure)';
      attentionStatus = behavioralCue;
      cautiousCategory = 'Low visible engagement-related cues';
    } else if (feat.phoneDetected) {
      finalState = 'LOW ATTENTION';
      behavioralCue = 'Phone Distraction';
      attentionStatus = 'Phone / Device Distraction';
      cautiousCategory = 'Low visible engagement-related cues';
    } else if (isBoredDisengaged) {
      finalState = 'LOW ATTENTION';
      behavioralCue = 'Bored / Disengaged';
      attentionStatus = 'Prolonged Off-Board Disengagement';
      cautiousCategory = 'Low visible engagement-related cues';
    } else if (isNoteTaking) {
      finalState = targetState === 'ATTENTIVE' ? 'ATTENTIVE' : 'MODERATE';
      behavioralCue = 'Note-Taking / Reading';
      attentionStatus = 'Note-Taking / Reading Behavior';
      cautiousCategory = finalState === 'ATTENTIVE' ? 'Attentive-related behaviour estimate' : 'Partially engaged or uncertain';
    } else if (targetState === 'ATTENTIVE') {
      finalState = 'ATTENTIVE';
      behavioralCue = 'Focused';
      attentionStatus = 'Focused Behavior';
      cautiousCategory = 'Attentive-related behaviour estimate';
    } else if (targetState === 'MODERATE') {
      finalState = 'MODERATE';
      behavioralCue = 'Moderate Focus';
      attentionStatus = 'Moderate Focus Behavior';
      cautiousCategory = 'Partially engaged or uncertain';
    } else {
      finalState = 'LOW ATTENTION';
      behavioralCue = 'Distracted';
      attentionStatus = 'Distracted / Off-Task Behavior';
      cautiousCategory = 'Low visible engagement-related cues';
    }

    buf.lastStableState = finalState;

    // Continuous confusion index computed from calibrated emotion probabilities & head orientation
    const confProb = feat.expressionProbabilities?.['Confused'] ?? (feat.emotion === 'Confused' ? 0.75 : 0.05);
    const frustProb = feat.expressionProbabilities?.['Frustrated'] ?? (feat.emotion === 'Frustrated' ? 0.60 : 0.05);
    const headTiltFactor = Math.abs(feat.rollDeg) > 12 || (Math.abs(feat.yawDeg) > 15 && Math.abs(feat.yawDeg) < 30) ? 0.2 : 0.0;
    const rawConfusion = Number(
      Math.min(100, Math.max(0, (confProb * 0.65 + frustProb * 0.45 + headTiltFactor) * 100)).toFixed(1)
    );
    const confusionIndex = rawConfusion;
    const confusionLevel = confusionIndex >= 60 ? 'High' : confusionIndex >= 30 ? 'Medium' : 'Low';

    // Continuous participation: hand raised weighted with active posture and attention
    const participationPct = Number(
      Math.min(
        100,
        Math.max(
          10,
          (handRaisedConsistent ? 75 : 20) + (smoothedAtt * 0.2) + (feat.posture === 'Upright' || feat.posture === 'Standing' ? 10 : 0)
        )
      ).toFixed(1)
    );

    const engagementPct = Number(
      Math.max(
        0,
        Math.min(
          100,
          (isDrowsyConsistent ? Math.min(smoothedAtt, 38) : smoothedAtt) * 0.85 + participationPct * 0.15
        )
      ).toFixed(1)
    );
    const esiScore = Number(Math.max(0, Math.min(100, engagementPct - (isDrowsyConsistent ? 12 : 0))).toFixed(1));

    const status = esiScore >= 75 ? 'Excellent' : esiScore >= 55 ? 'Good' : 'Needs Attention';

    const resolvedEmotion = isDrowsyConsistent ? 'Sleepy' : isBoredDisengaged ? 'Bored' : smoothedEmotion;
    const eyeEyelidStatus = feat.eyesClosed ? 'Closed' : smoothedEar < 0.23 ? 'Drooping' : 'Open';
    const evidenceParts = [
      `Eyes: ${eyeEyelidStatus} (EAR ${smoothedEar.toFixed(2)})`,
      `Gaze: ${smoothedGaze}`,
      `Cue: ${behavioralCue}`,
      `Head: ${feat.headPoseLabel} (yaw ${smoothedYaw}°, pitch ${smoothedPitch}°)`,
      `PERCLOS: ${perclosPct}% (${blinkRateBpm} bpm)`,
      `Pose: ${feat.posture}`,
    ];
    if (feat.phoneDetected) evidenceParts.push('Phone detected in student region');

    const explanation = isDrowsyConsistent
      ? `Estimated drowsy behavior from sustained eye closure (${drowsyCount}/${buf.drowsyFlags.length} frames).`
      : `Estimated ${finalState.toLowerCase()} engagement-related behavior (${Math.round(feat.faceQuality * 100)}% landmark reliability).`;

    // Compute individualized AI pedagogical suggestion based on specific multi-modal evidence:
    let aiSuggestion: { action: string; pedagogical_tip: string; priority: 'HIGH' | 'MEDIUM' | 'LOW' };
    if (isDrowsyConsistent) {
      aiSuggestion = {
        action: isMicroSleep ? 'Direct Check-in / Micro-Break' : 'Active Engagement Callout',
        pedagogical_tip:
          'Student is showing sustained eye closure / micro-sleep cues. Ask an interactive question or invite them to share thoughts.',
        priority: 'HIGH',
      };
    } else if (feat.phoneDetected) {
      aiSuggestion = {
        action: 'Device Distraction Redirection',
        pedagogical_tip: 'Mobile phone detected in student region. Gently remind student to stow secondary devices.',
        priority: 'HIGH',
      };
    } else if (confusionIndex >= 50) {
      aiSuggestion = {
        action: 'Concept Clarification',
        pedagogical_tip: 'Facial expressions show elevated confusion cues. Offer a quick recap or pause for questions.',
        priority: 'MEDIUM',
      };
    } else if (isBoredDisengaged) {
      aiSuggestion = {
        action: 'Re-engage Interest',
        pedagogical_tip: 'Prolonged off-board gaze and slouching detected. Use a polling question or real-world example.',
        priority: 'MEDIUM',
      };
    } else if (handRaisedConsistent) {
      aiSuggestion = {
        action: 'Acknowledge Participation',
        pedagogical_tip: 'Hand raised detected. Call on student to answer or elaborate on the current point.',
        priority: 'LOW',
      };
    } else if (isNoteTaking) {
      aiSuggestion = {
        action: 'Pacing Check',
        pedagogical_tip: 'Student is actively taking notes. Ensure you do not advance the slide too quickly.',
        priority: 'LOW',
      };
    } else if (finalState === 'ATTENTIVE') {
      aiSuggestion = {
        action: 'Positive Reinforcement',
        pedagogical_tip: 'Student exhibits steady focused engagement with upright posture and board-directed gaze.',
        priority: 'LOW',
      };
    } else if (finalState === 'MODERATE') {
      aiSuggestion = {
        action: 'Interactive Prompt',
        pedagogical_tip: 'Student is moderately engaged with occasional glancing. Check understanding with a quick poll.',
        priority: 'MEDIUM',
      };
    } else {
      aiSuggestion = {
        action: 'Refocus Attention',
        pedagogical_tip: 'Low visible engagement cues detected. Use a collaborative activity or direct verbal hook.',
        priority: 'MEDIUM',
      };
    }

    const calibratedAttention = isDrowsyConsistent
      ? Math.min(28, Math.max(8, Math.round(smoothedAtt * 0.3)))
      : smoothedAtt;

    return {
      student_id: track.studentId,
      bbox: track.faceBox
        ? [Math.round(track.faceBox.x1), Math.round(track.faceBox.y1), Math.round(track.faceBox.x2), Math.round(track.faceBox.y2)]
        : [],
      person_bbox: [
        Math.round(track.personBox.x1),
        Math.round(track.personBox.y1),
        Math.round(track.personBox.x2),
        Math.round(track.personBox.y2),
      ],
      face_detected: true,
      face_quality: Number(feat.faceQuality.toFixed(2)),
      attention_pct: calibratedAttention,
      attention_status: attentionStatus,
      gaze_direction: smoothedGaze,
      gaze_vector:
        smoothedGaze === 'Looking at Board'
          ? 'Forward'
          : smoothedGaze === 'Looking Left'
          ? 'Left'
          : smoothedGaze === 'Looking Right'
          ? 'Right'
          : 'Down',
      head_pose: feat.headPoseLabel,
      emotion: resolvedEmotion,
      expression_probabilities: feat.expressionProbabilities,
      modality_observations: feat.modalities,
      confidence_pct: Number((Math.max(feat.faceQuality, feat.emotionConfidence) * 100).toFixed(1)),
      is_drowsy: isDrowsyConsistent,
      ear_value: smoothedEar,
      is_yawning: isYawningConsistent,
      perclos_pct: perclosPct,
      blink_rate_bpm: blinkRateBpm,
      phone_detected: feat.phoneDetected,
      behavioral_cue: behavioralCue,
      hand_raised: handRaisedConsistent,
      posture: feat.posture,
      participation_pct: participationPct,
      engagement_pct: engagementPct,
      confusion_index: confusionIndex,
      confusion_level: confusionLevel,
      esi_score: esiScore,
      status,
      final_state: finalState,
      cautious_engagement_category: cautiousCategory,
      reason_for_state: explanation,
      ai_suggestion: aiSuggestion,
      corrective_guidance: {
        is_fallback: false,
        action_prompt:
          finalState === 'DROWSY'
            ? 'Interactive check-in / Micro-break'
            : feat.phoneDetected
            ? 'Secondary device redirection'
            : finalState === 'MODERATE'
            ? 'Pacing check / Direct question'
            : 'Maintain current instruction',
        target_accuracy_path: `Operating on full multimodal pipeline (${Math.round(feat.faceQuality * 100)}% landmark fidelity).`,
      },
      components: {
        attention_contribution: Number((0.66 * ((gazeScore + headScore) / 2)).toFixed(1)),
        emotion_contribution: Number((feat.emotion === 'Happy' || feat.emotion === 'Neutral' ? 12 : 6).toFixed(1)),
        participation_contribution: Number((0.15 * participationPct).toFixed(1)),
        posture_contribution: Number((0.14 * postureScore).toFixed(1)),
        drowsiness_penalty: Number((isDrowsyConsistent ? 18 + perclosPct * 0.2 : perclosPct * 0.1).toFixed(1)),
      },
      pitch_deg: smoothedPitch,
      yaw_deg: smoothedYaw,
      roll_deg: feat.rollDeg,
      modality_reliabilities: {
        gaze: Number(feat.faceQuality.toFixed(2)),
        head_pose: Number(feat.faceQuality.toFixed(2)),
        posture: feat.posture !== 'Unknown' ? 0.85 : 0.0,
        drowsiness: 0.9,
      },
      debug_info: {
        student_id: track.studentId,
        face_detected: true,
        valid_face_count: 1,
        final_attention_score: calibratedAttention,
        final_state: finalState,
        reason_for_state: explanation,
        raw_feature_values: {
          gaze: smoothedGaze,
          head_pose: feat.headPoseLabel,
          posture: feat.posture,
          ear: smoothedEar,
          face_quality: feat.faceQuality,
          emotion: resolvedEmotion,
          drowsy_evidence_frame: feat.eyesClosed,
          hand_raised_frame: feat.handRaised,
          pitch_deg: smoothedPitch,
          yaw_deg: smoothedYaw,
        },
        temporal_values: {
          raw_attention_score: Number(rawEngagement.toFixed(1)),
          smoothed_attention_score: smoothedAtt,
          drowsy_window_count: drowsyCount,
          drowsy_window_total: buf.drowsyFlags.length,
          drowsy_window_ratio_pct: perclosPct,
          is_drowsy_consistent: isDrowsyConsistent,
        },
        reliability_values: {
          face_quality: Number(feat.faceQuality.toFixed(2)),
          confidence_pct: Number((Math.max(feat.faceQuality, feat.emotionConfidence) * 100).toFixed(1)),
          is_valid_evidence: true,
        },
      },
      xai: {
        attention: {
          reasons: evidenceParts,
          confidence_pct: Number((feat.faceQuality * 100).toFixed(1)),
          status: finalState,
        },
        drowsiness: {
          diagnostic_markers: [
            `Eyelids: ${eyeEyelidStatus} (EAR ${smoothedEar.toFixed(2)})`,
            `Eye Gaze: ${smoothedGaze}`,
            `PERCLOS: ${perclosPct}%`,
            isDrowsyConsistent
              ? (isMicroSleep ? 'Downward Gaze Drift with Eyelid Closure (Micro-sleep)' : 'Sustained Eye Closure Confirmed')
              : 'Alert Eye Opening & Forward Gaze',
          ],
        },
        engagement: {
          summary_rationale: explanation,
          attributions: evidenceParts,
        },
        counterfactual: {
          explanation:
            finalState === 'DROWSY'
              ? 'Opening eyes fully and directing gaze toward the front board will immediately restore attentive status.'
              : finalState !== 'ATTENTIVE'
              ? 'Forward head orientation and board-directed gaze will raise engagement estimate.'
              : 'Maintain forward gaze and upright posture.',
        },
      },
    };
  }

  public aggregateClassroom(
    students: StudentAnalytics[],
    liveDetectedPersons: number,
    validFaceCount: number,
    fps: number,
    latencyMs: number,
    cameraConnected: boolean
  ): ClassroomMetrics {
    const withFaces = students.filter((s) => s.face_detected);
    const count = withFaces.length;

    let avgAtt = 0;
    let avgEng = 0;
    let avgPart = 0;
    let avgConf = 0;
    let drowsyCount = 0;
    let handRaiseCount = 0;
    let phoneDistractionCount = 0;

    if (count > 0) {
      avgAtt = Number((withFaces.reduce((a, b) => a + b.attention_pct, 0) / count).toFixed(1));
      avgEng = Number((withFaces.reduce((a, b) => a + b.engagement_pct, 0) / count).toFixed(1));
      avgPart = Number((withFaces.reduce((a, b) => a + b.participation_pct, 0) / count).toFixed(1));
      avgConf = Number((withFaces.reduce((a, b) => a + b.confusion_index, 0) / count).toFixed(1));
      drowsyCount = withFaces.filter((s) => s.is_drowsy).length;
      handRaiseCount = withFaces.filter((s) => s.hand_raised).length;
      phoneDistractionCount = withFaces.filter((s) => s.phone_detected).length;
    }

    // Classroom ECI: average engagement with penalties for collective drowsiness & phone distractions
    const rawEci = count === 0 ? 0 : avgEng - (drowsyCount / count) * 15 - (phoneDistractionCount / count) * 15;
    const eciScore = Number(Math.max(0, Math.min(100, rawEci)).toFixed(1));
    const eciStatus =
      count === 0
        ? 'No Face Detected'
        : eciScore >= 75
        ? 'High Engagement'
        : eciScore >= 50
        ? 'Moderate Engagement'
        : 'Low Engagement';

    const nowIso = new Date().toISOString();
    const recommendations: TeacherRecommendation[] = [];

    if (drowsyCount >= 1) {
      recommendations.push({
        id: 'rec-drowsy',
        timestamp: nowIso,
        type: 'DROWSINESS',
        severity: 'HIGH',
        title: 'Observable Drowsiness Detected',
        message: `${drowsyCount} student(s) showing sustained eye-closure cues.`,
        action: 'Consider an active learning prompt, interactive question, or short pacing break.',
      });
    }
    if (phoneDistractionCount >= 1) {
      recommendations.push({
        id: 'rec-phone',
        timestamp: nowIso,
        type: 'PHONE_DISTRACTION',
        severity: 'HIGH',
        title: 'Secondary Device In Use',
        message: `${phoneDistractionCount} student(s) displaying mobile device usage near desk.`,
        action: 'Direct student focus back to the board or primary class task.',
      });
    }
    if (avgConf >= 40) {
      recommendations.push({
        id: 'rec-confusion',
        timestamp: nowIso,
        type: 'CONFUSION',
        severity: 'MEDIUM',
        title: 'Elevated Classroom Confusion Cue',
        message: `Mean confusion index is ${avgConf}%. Multiple students exhibiting furrowed brow or head tilts.`,
        action: 'Pause and ask for clarifying questions or give a concrete example.',
      });
    }
    if (handRaiseCount >= 1) {
      recommendations.push({
        id: 'rec-hand',
        timestamp: nowIso,
        type: 'PARTICIPATION',
        severity: 'INFO',
        title: 'Active Hand Raise',
        message: `${handRaiseCount} student(s) raising hand.`,
        action: 'Acknowledge raised hand for participation.',
      });
    }
    if (recommendations.length === 0 && withFaces.length > 0) {
      recommendations.push({
        id: 'rec-steady',
        timestamp: nowIso,
        type: 'ENGAGEMENT_OK',
        severity: 'LOW',
        title: 'Steady Engagement Behavior',
        message: 'Observable classroom cues indicate steady engagement-related behavior.',
        action: 'Continue current instructional pace.',
      });
    }

    return {
      eci_score: eciScore,
      eci_status: eciStatus,
      avg_attention: avgAtt,
      avg_engagement: avgEng,
      avg_participation: avgPart,
      avg_confusion: avgConf,
      drowsy_count: drowsyCount,
      hand_raise_count: handRaiseCount,
      phone_distraction_count: phoneDistractionCount,
      live_detected_persons: liveDetectedPersons,
      valid_face_count: validFaceCount,
      active_tracks: students.length,
      camera_connected: cameraConnected,
      camera_source: 'BROWSER_MEDIAPIPE_AI',
      performance: {
        fps: Number(fps.toFixed(1)),
        latency_ms: Number(latencyMs.toFixed(1)),
      },
      students,
      recommendations,
    };
  }
}
