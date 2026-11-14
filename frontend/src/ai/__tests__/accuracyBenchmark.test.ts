import { describe, expect, it } from 'vitest';
import {
  applyNms,
  isFaceInPersonHeadRegion,
  isInsideDistractorScreen,
  isPhoneAssociatedWithPerson,
  NormalizedLandmark,
  suppressDuplicateFaces,
  validateFacialLandmarks,
} from '../validation';
import { BrowserByteTracker } from '../tracker';
import {
  classifyExpressionFromBlendshapes,
  estimateGazeDirection,
  estimateHeadPoseFromLandmarks,
  estimateHeadPoseFromMatrix,
  extractBehavioralFeatures,
  extractPoseBehavior,
} from '../features';
import { BrowserEngagementFusion } from '../fusion';

function makeLandmarks(foreheadY = 0.16, noseY = 0.27, chinY = 0.35): NormalizedLandmark[] {
  const lms: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.3 }));
  lms[10] = { x: 0.5, y: foreheadY };
  lms[33] = { x: 0.44, y: 0.22 };
  lms[263] = { x: 0.56, y: 0.22 };
  lms[1] = { x: 0.5, y: noseY };
  lms[152] = { x: 0.5, y: chinY };
  return lms;
}

function makePose(shoulderYNorm = 0.42): NormalizedLandmark[] {
  const pose: NormalizedLandmark[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
  pose[0] = { x: 0.5, y: 0.25, visibility: 0.9 }; // nose
  pose[11] = { x: 0.42, y: shoulderYNorm, visibility: 0.9 }; // left shoulder
  pose[12] = { x: 0.58, y: shoulderYNorm, visibility: 0.9 }; // right shoulder
  pose[13] = { x: 0.40, y: 0.50, visibility: 0.85 }; // left elbow
  pose[14] = { x: 0.60, y: 0.50, visibility: 0.85 }; // right elbow
  pose[15] = { x: 0.40, y: 0.60, visibility: 0.85 }; // left wrist
  pose[16] = { x: 0.60, y: 0.60, visibility: 0.85 }; // right wrist
  pose[23] = { x: 0.45, y: 0.75, visibility: 0.85 }; // left hip
  pose[24] = { x: 0.55, y: 0.75, visibility: 0.85 }; // right hip
  return pose;
}

describe('Stage 8: Comprehensive Accuracy & Robustness Benchmark Suite', () => {
  it('Scenario 1: Empty Classroom Condition -> Precision = 1.0, 0 False Positives', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const tracks = tracker.update([]);
    expect(tracks.length).toBe(0);

    const agg = fusion.aggregateClassroom([], 0, 0, 10.0, 12.0, true);
    expect(agg.live_detected_persons).toBe(0);
    expect(agg.valid_face_count).toBe(0);
    expect(agg.eci_score).toBe(0);
    expect(agg.eci_status).toBe('No Face Detected');
  });

  it('Scenario 2: Single Student Frontal Condition -> Valid Track & Attentive Estimate', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const p1 = { x1: 200, y1: 80, x2: 440, y2: 440 };
    const f1 = { x1: 280, y1: 100, x2: 360, y2: 190 };
    const lms = makeLandmarks();

    const tracks = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.92, faceConfidence: 0.89, qualityScore: 0.88 },
    ]);
    expect(tracks).toHaveLength(1);
    expect(tracks[0].studentId).toBe('Student 01');

    const feat = extractBehavioralFeatures(lms, [], makePose(), 0.88, 640, 480);
    const result = fusion.evaluateStudent(tracks[0], feat);
    expect(result.face_detected).toBe(true);
    expect(result.cautious_engagement_category).toBe('Attentive-related behaviour estimate');
  });

  it('Scenario 3: Multiple Seated Students (Crowded 3-Student Classroom) -> Separate Stable Tracks', () => {
    const tracker = new BrowserByteTracker(15, 0.25, 2);
    const p1 = { x1: 40, y1: 80, x2: 180, y2: 420 };
    const f1 = { x1: 80, y1: 100, x2: 140, y2: 170 };
    const p2 = { x1: 240, y1: 80, x2: 380, y2: 420 };
    const f2 = { x1: 280, y1: 100, x2: 340, y2: 170 };
    const p3 = { x1: 440, y1: 80, x2: 580, y2: 420 };
    const f3 = { x1: 480, y1: 100, x2: 540, y2: 170 };

    // Two frames to confirm tracks
    tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 },
      { personBox: p2, faceBox: f2, confidence: 0.88, faceConfidence: 0.87, qualityScore: 0.84 },
      { personBox: p3, faceBox: f3, confidence: 0.91, faceConfidence: 0.9, qualityScore: 0.86 },
    ]);
    const confirmed = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 },
      { personBox: p2, faceBox: f2, confidence: 0.88, faceConfidence: 0.87, qualityScore: 0.84 },
      { personBox: p3, faceBox: f3, confidence: 0.91, faceConfidence: 0.9, qualityScore: 0.86 },
    ]);

    expect(confirmed.map((t) => t.studentId)).toEqual(['Student 01', 'Student 02', 'Student 03']);
    expect(confirmed.every((t) => t.lifecycleState === 'confirmed')).toBe(true);
  });

  it('Scenario 4: Face Turned Away Condition -> Person Track Preserved, Face Marked Occluded', () => {
    const tracker = new BrowserByteTracker();
    const fusion = new BrowserEngagementFusion();
    const p1 = { x1: 150, y1: 80, x2: 350, y2: 420 };

    // Person tracked with face turned away (no face detection returned)
    const tracks = tracker.update([
      { personBox: p1, faceBox: null, confidence: 0.88, faceConfidence: 0, qualityScore: 0 },
    ]);
    expect(tracks[0].hasValidFace).toBe(false);
    expect(tracks[0].faceBox).toBeNull();

    const feat = extractBehavioralFeatures(null, undefined, makePose(), 0, 640, 480);
    const evalRes = fusion.evaluateStudent(tracks[0], feat);
    expect(evalRes.face_detected).toBe(false);
    expect(evalRes.final_state).toBe('UNABLE TO DETERMINE');
    expect(evalRes.cautious_engagement_category).toBe('Insufficient evidence');
    expect(evalRes.reason_for_state).toContain('cues occluded or turned away');
  });

  it('Scenario 5: Partial Occlusion & Recovery (Track Continuity across Missing Frames)', () => {
    const tracker = new BrowserByteTracker(10, 0.25, 2);
    const p1 = { x1: 150, y1: 80, x2: 350, y2: 420 };
    const f1 = { x1: 210, y1: 100, x2: 290, y2: 180 };

    tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 }]);
    tracker.update([{ personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 }]);

    // Occluded for 3 frames
    tracker.update([]);
    tracker.update([]);
    const occluded = tracker.update([]);
    expect(occluded[0].lifecycleState).toBe('occluded');
    expect(occluded[0].visibleThisFrame).toBe(false);

    // Re-appears: track identity must be preserved (IDF1 continuity)
    const recovered = tracker.update([
      { personBox: p1, faceBox: f1, confidence: 0.9, faceConfidence: 0.88, qualityScore: 0.85 },
    ]);
    expect(recovered[0].studentId).toBe('Student 01');
    expect(recovered[0].lifecycleState).toBe('confirmed');
    expect(recovered[0].missingFrames).toBe(0);
  });

  it('Scenario 6: Distractor Rejection (T-Shirt Torso Photo, Wall Poster, and Laptop Screen Face)', () => {
    const person = { x1: 180, y1: 50, x2: 460, y2: 450 };
    const pose = makePose(0.40); // shoulders around y = 192px

    // A. T-shirt print (face in lower torso, below shoulder line)
    const tshirtFace = { x1: 270, y1: 240, x2: 360, y2: 330 };
    const tshirtCheck = isFaceInPersonHeadRegion(tshirtFace, person, pose, 640, 480);
    expect(tshirtCheck.valid).toBe(false);
    expect(tshirtCheck.reason).toContain('T-shirt/torso rejection');

    // B. Laptop screen face (enclosed inside detected laptop distractor box)
    const laptopBox = { x1: 240, y1: 200, x2: 420, y2: 360, score: 0.89, label: 'laptop' };
    const screenFace = { x1: 280, y1: 220, x2: 350, y2: 290 };
    expect(isInsideDistractorScreen(screenFace, [laptopBox])).toBe(true);

    // C. Non-screen real face is preserved
    const realHeadFace = { x1: 280, y1: 75, x2: 360, y2: 165 };
    expect(isInsideDistractorScreen(realHeadFace, [laptopBox])).toBe(false);
    expect(isFaceInPersonHeadRegion(realHeadFace, person, pose, 640, 480).valid).toBe(true);
  });

  it('Scenario 7: Duplicate Bounding Box Suppression & Containment Handling', () => {
    const rawBoxes = [
      { x1: 100, y1: 80, x2: 200, y2: 300, score: 0.95 },
      { x1: 105, y1: 82, x2: 198, y2: 295, score: 0.70 }, // high IoU duplicate
      { x1: 120, y1: 100, x2: 180, y2: 220, score: 0.65 }, // contained duplicate
      { x1: 300, y1: 80, x2: 400, y2: 300, score: 0.92 }, // distinct person
    ];
    const nms = applyNms(rawBoxes, 0.45);
    expect(nms).toHaveLength(2);
    expect(nms[0].score).toBe(0.95);
    expect(nms[1].score).toBe(0.92);
  });

  it('Scenario 8: Head Pose Estimation Verification (Matrix vs Landmark Geometric Fallback)', () => {
    // Identity rotation matrix -> Yaw: 0, Pitch: 0, Roll: 0, Label: Forward
    const identityMatrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const matrixRes = estimateHeadPoseFromMatrix(identityMatrix);
    expect(matrixRes.label).toBe('Forward');
    expect(Math.abs(matrixRes.yawDeg)).toBeLessThan(2);
    expect(Math.abs(matrixRes.pitchDeg)).toBeLessThan(2);

    // Geometric landmark fallback
    const lms = makeLandmarks(0.16, 0.27, 0.35);
    const lmRes = estimateHeadPoseFromLandmarks(lms, 640, 480);
    expect(lmRes.label).toBe('Forward');
    expect(Number.isFinite(lmRes.yawDeg)).toBe(true);
  });

  it('Scenario 9: Body Posture & Hand Gestures (Standing, Slouching, Hand Raise)', () => {
    // A. Hand Raised: wrist y significantly higher than shoulder y
    const poseRaise = makePose(0.45);
    poseRaise[15] = { x: 0.40, y: 0.20, visibility: 0.92 }; // left wrist raised
    const raiseRes = extractPoseBehavior(poseRaise);
    expect(raiseRes.handRaised).toBe(true);
    expect(raiseRes.handConfidence).toBeGreaterThan(0.7);

    // B. Slouching: shoulder tilt
    const poseSlouch = makePose(0.40);
    poseSlouch[11] = { x: 0.42, y: 0.35, visibility: 0.9 };
    poseSlouch[12] = { x: 0.58, y: 0.48, visibility: 0.9 }; // shoulder tilt > 0.07
    const slouchRes = extractPoseBehavior(poseSlouch);
    expect(slouchRes.posture).toBe('Slouching');
  });

  it('Scenario 10: Dynamic Student Ingress & Egress Lifecycle', () => {
    const tracker = new BrowserByteTracker(3, 0.25, 2);
    const p1 = { x1: 50, y1: 80, x2: 180, y2: 400 };
    const p2 = { x1: 300, y1: 80, x2: 430, y2: 400 };

    // Student 1 enters
    tracker.update([{ personBox: p1, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 }]);
    const frame2 = tracker.update([
      { personBox: p1, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 },
    ]);
    expect(frame2[0].studentId).toBe('Student 01');

    // Student 2 enters
    tracker.update([
      { personBox: p1, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 },
      { personBox: p2, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 },
    ]);
    const frame4 = tracker.update([
      { personBox: p1, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 },
      { personBox: p2, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 },
    ]);
    expect(frame4.map((t) => t.studentId)).toEqual(['Student 01', 'Student 02']);

    // Student 1 leaves (egress) -> after 3 missing frames, expired
    tracker.update([{ personBox: p2, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 }]);
    tracker.update([{ personBox: p2, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 }]);
    tracker.update([{ personBox: p2, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 }]);
    const afterEgress = tracker.update([
      { personBox: p2, faceBox: null, confidence: 0.9, faceConfidence: 0, qualityScore: 0 },
    ]);
    expect(afterEgress.map((t) => t.studentId)).toEqual(['Student 02']);
  });
});

