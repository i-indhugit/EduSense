import { describe, expect, it } from 'vitest';
import {
  isFaceInPersonHeadRegion,
  isInsideDistractorScreen,
  isPhoneAssociatedWithPerson,
  NormalizedLandmark,
  suppressDuplicateFaces,
  validateFacialLandmarks,
} from '../validation';
import { BrowserByteTracker } from '../tracker';
import {
  BehavioralFeatures,
  classifyExpressionFromBlendshapes,
  estimateGazeDirection,
  extractBehavioralFeatures,
} from '../features';
import { BrowserEngagementFusion } from '../fusion';

function makeValidFaceLandmarks(): NormalizedLandmark[] {
  const lms: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.3 }));
  lms[10] = { x: 0.5, y: 0.16 }; // forehead
  lms[33] = { x: 0.44, y: 0.22 }; // left eye
  lms[263] = { x: 0.56, y: 0.22 }; // right eye
  lms[1] = { x: 0.5, y: 0.27 }; // nose
  lms[152] = { x: 0.5, y: 0.35 }; // chin
  return lms;
}

function makePoseLandmarks(shoulderYNorm = 0.4): NormalizedLandmark[] {
  const pose: NormalizedLandmark[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
  pose[11] = { x: 0.42, y: shoulderYNorm, visibility: 0.9 };
  pose[12] = { x: 0.58, y: shoulderYNorm, visibility: 0.9 };
  return pose;
}

describe('Browser AI Detection Validation & Tracking', () => {
  it('1. Rejects empty frame / no people -> 0 tracks & 0 faces', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const tracks = tracker.update([]);
    expect(tracks).toHaveLength(0);
    const agg = fusion.aggregateClassroom([], 0, 0, 10, 15, true);
    expect(agg.live_detected_persons).toBe(0);
    expect(agg.valid_face_count).toBe(0);
    expect(agg.eci_score).toBe(0);
  });

  it('2. Validates face in upper person head region and rejects T-shirt torso print', () => {
    const person = { x1: 200, y1: 60, x2: 420, y2: 440 };
    const pose = makePoseLandmarks(0.42); // y = 201px in 480px height

    const realHeadFace = { x1: 270, y1: 85, x2: 350, y2: 175 };
    expect(isFaceInPersonHeadRegion(realHeadFace, person, pose, 640, 480).valid).toBe(true);

    const tshirtFace = { x1: 275, y1: 230, x2: 345, y2: 310 };
    const res = isFaceInPersonHeadRegion(tshirtFace, person, pose, 640, 480);
    expect(res.valid).toBe(false);
  });

  it('3. Rejects poster/screen faces enclosed inside distractor objects', () => {
    const screenFace = { x1: 480, y1: 120, x2: 550, y2: 200 };
    const tvBox = { x1: 450, y1: 90, x2: 610, y2: 260, score: 0.88, label: 'tv' };
    expect(isInsideDistractorScreen(screenFace, [tvBox])).toBe(true);
  });

  it('4. Validates 478-pt facial landmarks and rejects degenerate geometry', () => {
    const faceBox = { x1: 260, y1: 70, x2: 380, y2: 190 };
    const validLms = makeValidFaceLandmarks();
    const ok = validateFacialLandmarks(validLms, faceBox, 640, 480);
    expect(ok.valid).toBe(true);
    expect(ok.qualityScore).toBeGreaterThan(0.5);

    const missing = validateFacialLandmarks([], faceBox, 640, 480);
    expect(missing.valid).toBe(false);
  });

  it('5. Maintains stable monotonic IDs for multiple people & clears stale face boxes on occlusion', () => {
    const tracker = new BrowserByteTracker(3, 0.25);
    const p1 = { x1: 50, y1: 60, x2: 200, y2: 400 };
    const f1 = { x1: 95, y1: 80, x2: 155, y2: 150 };
    const p2 = { x1: 350, y1: 60, x2: 500, y2: 400 };
    const f2 = { x1: 395, y1: 80, x2: 455, y2: 150 };

    const t1 = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 },
      { personBox: p2, faceBox: f2, confidence: 0.88, faceConfidence: 0.84, qualityScore: 0.8 },
    ]);
    expect(t1.map((t) => t.studentId)).toEqual(['Student 01', 'Student 02']);

    // Frame 2: Student 02 temporarily occluded -> no stale faceBox, Student 01 preserved
    const t2 = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 },
    ]);
    const s2 = t2.find((t) => t.studentId === 'Student 02')!;
    expect(s2.visibleThisFrame).toBe(false);
    expect(s2.faceBox).toBeNull();

    // Expire Student 02 after > 3 missing frames
    tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 }]);
    tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 }]);
    const t5 = tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 }]);
    expect(t5.map((t) => t.studentId)).toEqual(['Student 01']);
  });

  it('6. Enforces contradiction prevention (DROWSY overrides ATTENTIVE) & handles occluded face', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const p1 = { x1: 100, y1: 60, x2: 300, y2: 420 };
    const f1 = { x1: 160, y1: 80, x2: 240, y2: 170 };

    const drowsyFeat: BehavioralFeatures = {
      faceDetected: true,
      headPoseLabel: 'Forward',
      pitchDeg: 2,
      yawDeg: 1,
      rollDeg: 0,
      gazeDirection: 'Looking at Board',
      earValue: 0.14,
      marValue: 0.1,
      eyesClosed: true,
      isYawning: false,
      eyeLookDownAvg: 0.1,
      gazeDeviation: 0.05,
      phoneDetected: false,
      emotion: 'Sleepy',
      emotionConfidence: 0.85,
      handRaised: false,
      posture: 'Upright',
      faceQuality: 0.82,
    };

    let lastEval;
    for (let i = 0; i < 5; i++) {
      const tracks = tracker.update([
        { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.9, qualityScore: 0.82 },
      ]);
      lastEval = fusion.evaluateStudent(tracks[0], drowsyFeat);
    }
    expect(lastEval?.final_state).toBe('DROWSY');
    expect(lastEval?.is_drowsy).toBe(true);

    // Occluded face -> UNABLE TO DETERMINE
    const tracksOcc = tracker.update([
      { personBox: p1, faceBox: null, confidence: 0.85, faceConfidence: 0, qualityScore: 0 },
    ]);
    const occEval = fusion.evaluateStudent(tracksOcc[0], { ...drowsyFeat, faceDetected: false, faceQuality: 0 });
    expect(occEval.final_state).toBe('UNABLE TO DETERMINE');
    expect(occEval.face_detected).toBe(false);
  });

  it('7. Computes PERCLOS, Phone Distraction, Note-Taking vs Micro-Sleep, and Boredom cues', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const p1 = { x1: 100, y1: 60, x2: 300, y2: 420 };
    const f1 = { x1: 160, y1: 80, x2: 240, y2: 170 };

    const baseFeat: BehavioralFeatures = {
      faceDetected: true,
      headPoseLabel: 'Looking Down',
      pitchDeg: 20,
      yawDeg: 0,
      rollDeg: 0,
      gazeDirection: 'Looking Down / Away',
      earValue: 0.28,
      marValue: 0.1,
      eyesClosed: false,
      isYawning: false,
      eyeLookDownAvg: 0.52,
      gazeDeviation: 0.2,
      phoneDetected: false,
      emotion: 'Neutral',
      emotionConfidence: 0.8,
      handRaised: false,
      posture: 'Upright',
      faceQuality: 0.85,
    };

    // Note-taking (eyes open, looking down)
    const trk1 = tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.9, qualityScore: 0.85 }]);
    const noteRes = fusion.evaluateStudent(trk1[0], baseFeat);
    expect(noteRes.behavioral_cue).toBe('Note-Taking / Reading');
    expect(noteRes.final_state).toBe('ATTENTIVE');

    // Phone distraction
    const phoneRes = fusion.evaluateStudent(trk1[0], { ...baseFeat, phoneDetected: true });
    expect(phoneRes.phone_detected).toBe(true);
    expect(phoneRes.behavioral_cue).toBe('Phone Distraction');
    expect(phoneRes.final_state).toBe('PHONE DISTRACTED');

    // Micro-sleep (pitchDeg > 18 + sustained eyesClosed) + PERCLOS
    let microRes;
    for (let i = 0; i < 4; i++) {
      microRes = fusion.evaluateStudent(trk1[0], { ...baseFeat, eyesClosed: true, earValue: 0.15 });
    }
    expect(microRes?.final_state).toBe('DROWSY');
    expect(microRes?.behavioral_cue).toBe('Micro-Sleep / Head Nodding');
    expect(microRes?.perclos_pct).toBeGreaterThan(30);

    // Boredom / Disengagement (prolonged off-board gaze + slouching)
    const trackerBored = new BrowserByteTracker();
    const fusionBored = new BrowserEngagementFusion();
    const boredFeat: BehavioralFeatures = {
      ...baseFeat,
      pitchDeg: 0,
      yawDeg: 28,
      headPoseLabel: 'Looking Right',
      gazeDirection: 'Looking Right',
      eyeLookDownAvg: 0.1,
      posture: 'Slouching',
    };
    let boredRes;
    const tb = trackerBored.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.9, qualityScore: 0.85 }]);
    for (let i = 0; i < 12; i++) {
      boredRes = fusionBored.evaluateStudent(tb[0], boredFeat);
    }
    expect(boredRes?.behavioral_cue).toBe('Bored / Disengaged');
    expect(boredRes?.emotion).toBe('Bored');
  });

  it('8. Suppresses duplicate overlapping faces & isolates phone distraction per student', () => {
    const dupFaces = suppressDuplicateFaces([
      { box: { x1: 100, y1: 80, x2: 180, y2: 170 }, score: 0.91 },
      { box: { x1: 104, y1: 82, x2: 182, y2: 172 }, score: 0.72 },
      { box: { x1: 320, y1: 80, x2: 400, y2: 170 }, score: 0.88 },
    ]);
    expect(dupFaces).toHaveLength(2);
    expect(dupFaces[0].score).toBe(0.91);

    const student1 = { x1: 60, y1: 60, x2: 220, y2: 420 };
    const student2 = { x1: 280, y1: 60, x2: 440, y2: 420 };
    const phoneNearStudent1 = [{ x1: 120, y1: 260, x2: 165, y2: 330, score: 0.75, label: 'cell phone' }];
    expect(isPhoneAssociatedWithPerson(student1, phoneNearStudent1)).toBe(true);
    expect(isPhoneAssociatedWithPerson(student2, phoneNearStudent1)).toBe(false);
  });

  it('9. Recovers student ID after brief occlusion & expires unconfirmed 1-frame noise rapidly', () => {
    const tracker = new BrowserByteTracker(15, 0.25, 2);
    const p1 = { x1: 60, y1: 60, x2: 200, y2: 420 };
    const f1 = { x1: 100, y1: 80, x2: 160, y2: 160 };
    const noise = { x1: 480, y1: 100, x2: 560, y2: 300 };

    // Frame 1: both tentative
    tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 },
      { personBox: noise, faceBox: null, confidence: 0.55, faceConfidence: 0, qualityScore: 0 },
    ]);
    // Frame 2: p1 confirmed (2 hits), noise disappears
    const f2 = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 },
    ]);
    expect(f2.find((t) => t.studentId === 'Student 01')?.lifecycleState).toBe('confirmed');
    expect(f2.find((t) => t.studentId === 'Student 02')?.lifecycleState).toBe('occluded');

    // Frame 3-5: noise stays missing > 2 frames -> expired; p1 occluded for 2 frames then recovers
    tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 }]);
    const f4 = tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 }]);
    const f5 = tracker.update([]); // p1 occluded 1 frame
    expect(f4.map((t) => t.studentId)).toEqual(['Student 01']);
    expect(f5[0].lifecycleState).toBe('occluded');

    const recovered = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.85, qualityScore: 0.8 },
    ]);
    expect(recovered[0].studentId).toBe('Student 01');
    expect(recovered[0].lifecycleState).toBe('confirmed');
    expect(recovered[0].trackConfidence).toBeGreaterThan(0.5);
  });

  it('10. Compensates gaze with head pose, emits calibrated expression probabilities & modality metadata', () => {
    const lms = makeValidFaceLandmarks();
    // Head turned slightly left (-18 deg) while eyes look right to compensate -> still Looking at Board
    const compensatedGaze = estimateGazeDirection(
      lms,
      -18,
      0,
      [
        { categoryName: 'eyeLookInLeft', score: 0.45 },
        { categoryName: 'eyeLookOutRight', score: 0.45 },
      ],
      0.85
    );
    expect(compensatedGaze).toBe('Looking at Board');

    // Low quality face -> Unable to Determine
    expect(estimateGazeDirection(lms, 0, 0, [], 0.2)).toBe('Unable to Determine');

    // Calibrated expression probabilities sum to ~1
    const expr = classifyExpressionFromBlendshapes(
      [
        { categoryName: 'mouthSmileLeft', score: 0.75 },
        { categoryName: 'mouthSmileRight', score: 0.75 },
      ],
      false
    );
    expect(expr.valid).toBe(true);
    expect(expr.emotion).toBe('Happy');
    const probSum = Object.values(expr.probabilities).reduce((a, b) => a + b, 0);
    expect(probSum).toBeGreaterThan(0.98);
    expect(probSum).toBeLessThan(1.02);

    // Missing blendshapes -> valid: false with rejectionReason
    const missingExpr = classifyExpressionFromBlendshapes(undefined, false);
    expect(missingExpr.valid).toBe(false);
    expect(missingExpr.rejectionReason).toBeTruthy();

    // Standardized modalities with shoulder/elbow/wrist hand-raise
    const pose = makePoseLandmarks(0.45);
    pose[13] = { x: 0.40, y: 0.42, visibility: 0.9 }; // left elbow
    pose[15] = { x: 0.40, y: 0.25, visibility: 0.9 }; // left wrist raised above shoulder
    const feat = extractBehavioralFeatures(lms, [{ categoryName: 'eyeBlinkLeft', score: 0.1 }], pose, 0.85, 640, 480);
    expect(feat.handRaised).toBe(true);
    expect(feat.modalities?.gaze.valid).toBe(true);
    expect(feat.modalities?.handRaise.observation).toBe('Raised');
    expect(feat.modalities?.handRaise.timestamp).toBeGreaterThan(0);
  });

  it('11. Fuses modalities with reliability weights and produces cautious categories', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const p1 = { x1: 80, y1: 60, x2: 240, y2: 400 };
    const f1 = { x1: 120, y1: 80, x2: 200, y2: 170 };

    const trk = tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 }]);
    const lms = makeValidFaceLandmarks();
    const feat = extractBehavioralFeatures(
      lms,
      [{ categoryName: 'mouthSmileLeft', score: 0.6 }],
      makePoseLandmarks(0.4),
      0.85,
      640,
      480
    );

    const evalRes = fusion.evaluateStudent(trk[0], feat);
    expect(evalRes.cautious_engagement_category).toBe('Attentive-related behaviour estimate');
    expect(evalRes.final_state).toBe('ATTENTIVE');
    expect(evalRes.confusion_index).toBeLessThan(40);
    expect(evalRes.participation_pct).toBeGreaterThan(0);
  });
});

