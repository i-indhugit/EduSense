import { describe, expect, it } from 'vitest';
import { BrowserEngagementFusion } from '../fusion';
import { BehavioralFeatures } from '../features';
import { BrowserByteTracker, TrackedPerson } from '../tracker';

function createMockTrack(studentId = 'Student 01', trackId = 1): TrackedPerson {
  return {
    trackId,
    studentId,
    personBox: { x1: 100, y1: 50, x2: 300, y2: 400 },
    faceBox: { x1: 160, y1: 70, x2: 240, y2: 170 },
    confidence: 0.92,
    faceConfidence: 0.89,
    qualityScore: 0.88,
    visibleThisFrame: true,
    hasValidFace: true,
    missingFrames: 0,
    hits: 5,
    ageFrames: 10,
    lifecycleState: 'confirmed' as const,
    trackConfidence: 0.92,
    candidateIndex: 0,
  };
}

function createBaseFeatures(overrides: Partial<BehavioralFeatures> = {}): BehavioralFeatures {
  return {
    faceDetected: true,
    headPoseLabel: 'Forward',
    pitchDeg: 0,
    yawDeg: 0,
    rollDeg: 0,
    gazeDirection: 'Looking at Board',
    earValue: 0.28,
    marValue: 0.12,
    eyesClosed: false,
    isYawning: false,
    eyeLookDownAvg: 0.1,
    gazeDeviation: 0.06,
    phoneDetected: false,
    emotion: 'Neutral',
    emotionConfidence: 0.85,
    handRaised: false,
    posture: 'Upright',
    faceQuality: 0.88,
    expressionProbabilities: {
      Neutral: 0.85,
      Happy: 0.05,
      Confused: 0.03,
      Sleepy: 0.02,
      Frustrated: 0.03,
      Bored: 0.02,
    },
    modalities: {
      gaze: { observation: 'Looking at Board', confidence: 0.88, valid: true, timestamp: Date.now(), rejectionReason: null },
      headPose: { observation: 'Forward', confidence: 0.88, valid: true, timestamp: Date.now(), rejectionReason: null },
      eyeClosure: { observation: 'Open', confidence: 0.88, valid: true, timestamp: Date.now(), rejectionReason: null },
      posture: { observation: 'Upright', confidence: 0.90, valid: true, timestamp: Date.now(), rejectionReason: null },
      expression: { observation: 'Neutral', confidence: 0.85, valid: true, timestamp: Date.now(), rejectionReason: null },
      handRaise: { observation: 'Lowered', confidence: 0.95, valid: true, timestamp: Date.now(), rejectionReason: null },
    },
    ...overrides,
  };
}

describe('Targeted State Accuracy & Student-Level AI Suggestions Suite', () => {
  it('1. ATTENTIVE State: Steady forward gaze & posture yields >=75% attention and Positive Reinforcement suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const feat = createBaseFeatures({ gazeDirection: 'Looking at Board', yawDeg: 2, pitchDeg: -1, posture: 'Upright' });

    let result = fusion.evaluateStudent(track, feat);
    for (let i = 0; i < 5; i++) {
      result = fusion.evaluateStudent(track, feat);
    }

    expect(result.final_state).toBe('ATTENTIVE');
    expect(result.attention_pct).toBeGreaterThanOrEqual(75);
    expect(result.cautious_engagement_category).toBe('Attentive-related behaviour estimate');
    expect(result.ai_suggestion).toBeDefined();
    expect(result.ai_suggestion!.action).toBe('Positive Reinforcement');
    expect(result.ai_suggestion!.priority).toBe('LOW');
  });

  it('2. MODERATE State: Glanced sideways (yaw ~24°) reliably classifies as MODERATE with Interactive Prompt suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const moderateFeat = createBaseFeatures({
      gazeDirection: 'Looking Left',
      gazeDeviation: 0.42,
      yawDeg: -24,
      headPoseLabel: 'Looking Left',
      pitchDeg: 4,
    });

    let result = fusion.evaluateStudent(track, moderateFeat);
    for (let i = 0; i < 7; i++) {
      result = fusion.evaluateStudent(track, moderateFeat);
    }

    expect(result.final_state).toBe('MODERATE');
    expect(result.attention_pct).toBeGreaterThanOrEqual(44);
    expect(result.attention_pct).toBeLessThanOrEqual(72);
    expect(result.cautious_engagement_category).toBe('Partially engaged or uncertain');
    expect(result.ai_suggestion).toBeDefined();
    expect(result.ai_suggestion!.action).toBe('Interactive Prompt');
    expect(result.ai_suggestion!.priority).toBe('MEDIUM');
  });

  it('3. DROWSY State: Triggers within 2 closed frames, overrides attentive, and latches without flicker', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const attentiveFeat = createBaseFeatures();

    // 4 frames of attentive
    for (let i = 0; i < 4; i++) {
      fusion.evaluateStudent(track, attentiveFeat);
    }

    // 2 consecutive closed eye frames
    const closedFeat = createBaseFeatures({ eyesClosed: true, earValue: 0.14, emotion: 'Sleepy' });
    fusion.evaluateStudent(track, closedFeat);
    const drowsyFrame2 = fusion.evaluateStudent(track, closedFeat);

    expect(drowsyFrame2.final_state).toBe('DROWSY');
    expect(drowsyFrame2.is_drowsy).toBe(true);
    expect(drowsyFrame2.ai_suggestion!.action).toBe('Active Engagement Callout');
    expect(drowsyFrame2.ai_suggestion!.priority).toBe('HIGH');

    // 1 frame of transient eye opening (e.g. tracking noise / momentary flicker) -> MUST REMAIN DROWSY due to latch
    const openFrameAfterDrowsy = fusion.evaluateStudent(track, attentiveFeat);
    expect(openFrameAfterDrowsy.final_state).toBe('DROWSY');
    expect(openFrameAfterDrowsy.is_drowsy).toBe(true);
  });

  it('4. Micro-Sleep: Closed eyes with downward head tilt triggers Micro-Sleep cue and Direct Check-in suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const microSleepFeat = createBaseFeatures({
      eyesClosed: true,
      earValue: 0.12,
      pitchDeg: 22,
      headPoseLabel: 'Looking Down',
      emotion: 'Sleepy',
    });

    let res = fusion.evaluateStudent(track, microSleepFeat);
    for (let i = 0; i < 3; i++) {
      res = fusion.evaluateStudent(track, microSleepFeat);
    }

    expect(res.final_state).toBe('DROWSY');
    expect(res.behavioral_cue).toContain('Micro-Sleep');
    expect(res.ai_suggestion!.action).toBe('Direct Check-in / Micro-Break');
    expect(res.ai_suggestion!.priority).toBe('HIGH');
  });

  it('5. Phone Distraction: Secondary device triggers LOW ATTENTION and Device Distraction Redirection suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const phoneFeat = createBaseFeatures({ phoneDetected: true });

    const res = fusion.evaluateStudent(track, phoneFeat);
    expect(res.final_state).toBe('LOW ATTENTION');
    expect(res.phone_detected).toBe(true);
    expect(res.behavioral_cue).toBe('Phone Distraction');
    expect(res.ai_suggestion!.action).toBe('Device Distraction Redirection');
    expect(res.ai_suggestion!.priority).toBe('HIGH');
  });

  it('6. Confusion Cue: Expression index >= 50 triggers Concept Clarification suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const confusedFeat = createBaseFeatures({
      emotion: 'Confused',
      expressionProbabilities: {
        Confused: 0.78,
        Frustrated: 0.15,
        Neutral: 0.05,
        Happy: 0.01,
        Sleepy: 0.01,
        Bored: 0.0,
      },
      rollDeg: 16,
    });

    const res = fusion.evaluateStudent(track, confusedFeat);
    expect(res.confusion_index).toBeGreaterThanOrEqual(50);
    expect(res.confusion_level).toBe('High');
    expect(res.ai_suggestion!.action).toBe('Concept Clarification');
    expect(res.ai_suggestion!.priority).toBe('MEDIUM');
  });

  it('7. Hand Raised: Active participation triggers Acknowledge Participation suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const handFeat = createBaseFeatures({ handRaised: true });

    let res = fusion.evaluateStudent(track, handFeat);
    for (let i = 0; i < 3; i++) {
      res = fusion.evaluateStudent(track, handFeat);
    }

    expect(res.hand_raised).toBe(true);
    expect(res.participation_pct).toBeGreaterThanOrEqual(70);
    expect(res.ai_suggestion!.action).toBe('Acknowledge Participation');
    expect(res.ai_suggestion!.priority).toBe('LOW');
  });

  it('8. Note-Taking: Looking down with open eyes triggers Note-Taking cue and Pacing Check suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();
    const noteFeat = createBaseFeatures({
      eyesClosed: false,
      earValue: 0.27,
      pitchDeg: 20,
      eyeLookDownAvg: 0.65,
      gazeDirection: 'Looking Down / Away',
    });

    const res = fusion.evaluateStudent(track, noteFeat);
    expect(res.behavioral_cue).toBe('Note-Taking / Reading');
    expect(res.ai_suggestion!.action).toBe('Pacing Check');
    expect(res.ai_suggestion!.priority).toBe('LOW');
  });

  it('9. Face Occluded: Yields UNABLE TO DETERMINE and Face Visibility Required suggestion', () => {
    const fusion = new BrowserEngagementFusion();
    const occludedTrack = { ...createMockTrack(), visibleThisFrame: false, hasValidFace: false };
    const res = fusion.evaluateStudent(occludedTrack, createBaseFeatures({ faceDetected: false, faceQuality: 0 }));

    expect(res.face_detected).toBe(false);
    expect(res.final_state).toBe('UNABLE TO DETERMINE');
    expect(res.cautious_engagement_category).toBe('Insufficient evidence');
    expect(res.ai_suggestion!.action).toBe('Face Visibility Required');
    expect(res.ai_suggestion!.priority).toBe('LOW');
  });

  it('10. Eye Gaze & Eyelid Open/Close Diagnostics: Downward gaze drift + eyelid droop triggers micro-sleep drowsiness', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack();

    // Student A: Eyes open, looking forward at board -> Not drowsy, attentive
    const alertFeat = createBaseFeatures({ eyesClosed: false, earValue: 0.30, gazeDirection: 'Looking at Board' });
    const alertRes = fusion.evaluateStudent(track, alertFeat);
    expect(alertRes.is_drowsy).toBe(false);
    expect(alertRes.xai?.drowsiness?.diagnostic_markers).toBeDefined();
    expect(alertRes.xai?.drowsiness?.diagnostic_markers?.[0]).toContain('Eyelids: Open');

    // Student B: Eyelids droop (EAR < 0.22) and gaze drifts downward without note-taking
    const droopGazeFeat = createBaseFeatures({
      eyesClosed: true,
      earValue: 0.17,
      eyeLookDownAvg: 0.62,
      gazeDirection: 'Looking Down / Away',
      pitchDeg: 16,
    });
    fusion.evaluateStudent(track, droopGazeFeat);
    const drowsyRes = fusion.evaluateStudent(track, droopGazeFeat);

    expect(drowsyRes.final_state).toBe('DROWSY');
    expect(drowsyRes.is_drowsy).toBe(true);
    expect(drowsyRes.behavioral_cue).toContain('Micro-Sleep');
    expect(drowsyRes.xai?.drowsiness?.diagnostic_markers?.[0]).toContain('Eyelids: Closed');
  });

  it('11. Student ID Stability and Attention Score Calibration: Student 01 reuses slot & drowsy attention drops < 30%', () => {
    // 1. Verify student numbering slot stability
    const tracker = new BrowserByteTracker(3, 0.25, 1);
    const p1 = { x1: 100, y1: 50, x2: 300, y2: 400 };
    const f1 = { x1: 160, y1: 70, x2: 240, y2: 170 };

    // Initial detection -> Student 01
    const t1 = tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 }]);
    expect(t1[0].studentId).toBe('Student 01');

    // Person steps away for 5 frames (exceeds maxMissingFrames = 3)
    tracker.update([]);
    tracker.update([]);
    tracker.update([]);
    tracker.update([]);
    const tGone = tracker.update([]);
    expect(tGone).toHaveLength(0);

    // Person returns -> must be Student 01 again (never incrementing to Student 02, 03...)
    const tReturn = tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 }]);
    expect(tReturn[0].studentId).toBe('Student 01');

    // 2. Verify attention score ranges for states
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack('Student 01', 1);

    // Alert / Attentive: attention_pct in 70-98% range
    const alertFeat = createBaseFeatures({ eyesClosed: false, earValue: 0.30, gazeDirection: 'Looking at Board' });
    const attentiveRes = fusion.evaluateStudent(track, alertFeat);
    expect(attentiveRes.final_state).toBe('ATTENTIVE');
    expect(attentiveRes.attention_pct).toBeGreaterThanOrEqual(70);
    expect(attentiveRes.debug_info?.final_attention_score).toBe(attentiveRes.attention_pct);

    // Moderate / Turned head: attention_pct in 40-68% range
    const moderateFusion = new BrowserEngagementFusion();
    const turnedFeat = createBaseFeatures({
      eyesClosed: false,
      earValue: 0.28,
      gazeDirection: 'Looking Left',
      gazeDeviation: 0.42,
      yawDeg: -24,
      headPoseLabel: 'Looking Left',
      pitchDeg: 4,
    });
    let modRes = moderateFusion.evaluateStudent(track, turnedFeat);
    for (let i = 0; i < 5; i++) {
      modRes = moderateFusion.evaluateStudent(track, turnedFeat);
    }
    expect(modRes.final_state).toBe('MODERATE');
    expect(modRes.attention_pct).toBeGreaterThanOrEqual(40);
    expect(modRes.attention_pct).toBeLessThanOrEqual(68);

    // Drowsy: attention_pct calibrated to low range (15-28%)
    const drowsyFusion = new BrowserEngagementFusion();
    const drowsyFeat = createBaseFeatures({
      eyesClosed: true,
      earValue: 0.16,
      gazeDirection: 'Looking Down / Away',
      eyeLookDownAvg: 0.60,
    });
    drowsyFusion.evaluateStudent(track, drowsyFeat);
    const drowsyRes = drowsyFusion.evaluateStudent(track, drowsyFeat);
    expect(drowsyRes.final_state).toBe('DROWSY');
    expect(drowsyRes.attention_pct).toBeLessThanOrEqual(28);
    expect(drowsyRes.attention_pct).toBeGreaterThanOrEqual(8);
    expect(drowsyRes.debug_info?.final_attention_score).toBe(drowsyRes.attention_pct);
  });
});
