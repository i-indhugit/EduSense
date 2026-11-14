import { NormalizedLandmark } from './validation';

export interface BlendshapeCategory {
  categoryName: string;
  score: number;
}

export interface ModalityObservation<T = string> {
  observation: T;
  confidence: number;
  valid: boolean;
  timestamp: number;
  rejectionReason: string | null;
}

export interface BehavioralFeatures {
  faceDetected: boolean;
  headPoseLabel: 'Forward' | 'Looking Left' | 'Looking Right' | 'Looking Up' | 'Looking Down' | 'Unknown';
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
  gazeDirection: 'Looking at Board' | 'Looking Left' | 'Looking Right' | 'Looking Down / Away' | 'Unable to Determine';
  earValue: number;
  marValue: number;
  eyesClosed: boolean;
  isYawning: boolean;
  eyeLookDownAvg: number;
  gazeDeviation: number;
  phoneDetected: boolean;
  screenGlowDetected?: boolean;
  screenGlowConfidence?: number;
  emotion: 'Neutral' | 'Happy' | 'Confused' | 'Sleepy' | 'Frustrated' | 'Bored' | 'Unable to Determine';
  emotionConfidence: number;
  expressionProbabilities?: Record<string, number>;
  handRaised: boolean;
  posture: 'Upright' | 'Leaning' | 'Slouching' | 'Standing' | 'Unknown';
  faceQuality: number;
  modalities?: {
    gaze: ModalityObservation;
    headPose: ModalityObservation;
    expression: ModalityObservation;
    posture: ModalityObservation;
    handRaise: ModalityObservation;
    eyeClosure: ModalityObservation;
  };
}

function dist2D(a: NormalizedLandmark, b: NormalizedLandmark, w: number, h: number): number {
  return Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
}

function getBlendshapeScore(blendshapes: BlendshapeCategory[] | undefined, name: string): number {
  if (!blendshapes) return 0;
  const found = blendshapes.find((b) => b.categoryName === name);
  return found ? found.score : 0;
}

export function estimateHeadPoseFromMatrix(
  matrixData: number[] | Float32Array
): { yawDeg: number; pitchDeg: number; rollDeg: number; label: BehavioralFeatures['headPoseLabel'] } {
  // MediaPipe 4x4 column-major rotation matrix R:
  // m[0] m[4] m[8]
  // m[1] m[5] m[9]
  // m[2] m[6] m[10]
  const r00 = matrixData[0];
  const r10 = matrixData[1];
  const r20 = matrixData[2];
  const r21 = matrixData[6];
  const r22 = matrixData[10];

  const sy = Math.sqrt(r00 * r00 + r10 * r10);
  let pitchRad: number;
  let yawRad: number;
  let rollRad: number;

  if (sy > 1e-6) {
    pitchRad = Math.atan2(r21, r22);
    yawRad = Math.atan2(-r20, sy);
    rollRad = Math.atan2(r10, r00);
  } else {
    pitchRad = Math.atan2(-matrixData[9], matrixData[5]);
    yawRad = Math.atan2(-r20, sy);
    rollRad = 0;
  }

  // Convert to intuitive degrees (pitch > 0 looking down, yaw > 0 looking right)
  const pitchDeg = Number(Math.max(-50, Math.min(50, (-pitchRad * 180) / Math.PI)).toFixed(1));
  const yawDeg = Number(Math.max(-60, Math.min(60, (yawRad * 180) / Math.PI)).toFixed(1));
  const rollDeg = Number(Math.max(-45, Math.min(45, (rollRad * 180) / Math.PI)).toFixed(1));

  let label: BehavioralFeatures['headPoseLabel'] = 'Forward';
  if (yawDeg < -22) label = 'Looking Left';
  else if (yawDeg > 22) label = 'Looking Right';
  else if (pitchDeg > 20) label = 'Looking Down';
  else if (pitchDeg < -22) label = 'Looking Up';

  return { yawDeg, pitchDeg, rollDeg, label };
}

export function estimateHeadPoseFromLandmarks(
  landmarks: NormalizedLandmark[],
  imgW: number,
  imgH: number
): { yawDeg: number; pitchDeg: number; rollDeg: number; label: BehavioralFeatures['headPoseLabel'] } {
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];
  const nose = landmarks[1];
  const chin = landmarks[152];
  const forehead = landmarks[10];

  const lx = leftEye.x * imgW;
  const ly = leftEye.y * imgH;
  const rx = rightEye.x * imgW;
  const ry = rightEye.y * imgH;
  const nx = nose.x * imgW;
  const ny = nose.y * imgH;

  const eyeMidX = (lx + rx) / 2;
  const eyeMidY = (ly + ry) / 2;
  const eyeSpan = Math.max(1, Math.hypot(rx - lx, ry - ly));
  const faceHeight = Math.max(1, (chin.y - forehead.y) * imgH);

  // Geometric yaw/pitch/roll approximations calibrated to degrees
  const yawDeg = Math.max(-55, Math.min(55, ((nx - eyeMidX) / eyeSpan) * 75));
  const expectedNoseY = eyeMidY + faceHeight * 0.22;
  const pitchDeg = Math.max(-45, Math.min(45, ((ny - expectedNoseY) / faceHeight) * 95));
  const rollDeg = Math.max(-45, Math.min(45, (Math.atan2(ry - ly, rx - lx) * 180) / Math.PI));

  let label: BehavioralFeatures['headPoseLabel'] = 'Forward';
  if (yawDeg < -24) label = 'Looking Left';
  else if (yawDeg > 24) label = 'Looking Right';
  else if (pitchDeg > 22) label = 'Looking Down';
  else if (pitchDeg < -24) label = 'Looking Up';

  return {
    yawDeg: Number(yawDeg.toFixed(1)),
    pitchDeg: Number(pitchDeg.toFixed(1)),
    rollDeg: Number(rollDeg.toFixed(1)),
    label,
  };
}

export function estimateEyeAndMouth(
  landmarks: NormalizedLandmark[],
  imgW: number,
  imgH: number,
  blendshapes?: BlendshapeCategory[]
): { earValue: number; marValue: number; eyesClosed: boolean; isYawning: boolean } {
  // Left eye: [33, 160, 158, 133, 153, 144]
  const lV1 = dist2D(landmarks[160], landmarks[144], imgW, imgH);
  const lV2 = dist2D(landmarks[158], landmarks[153], imgW, imgH);
  const lH = Math.max(1, dist2D(landmarks[33], landmarks[133], imgW, imgH));
  const leftEar = (lV1 + lV2) / (2 * lH);

  // Right eye: [362, 385, 387, 263, 373, 380]
  const rV1 = dist2D(landmarks[385], landmarks[380], imgW, imgH);
  const rV2 = dist2D(landmarks[387], landmarks[373], imgW, imgH);
  const rH = Math.max(1, dist2D(landmarks[362], landmarks[263], imgW, imgH));
  const rightEar = (rV1 + rV2) / (2 * rH);

  const earValue = Number(((leftEar + rightEar) / 2).toFixed(3));

  // Mouth: [13, 14] vertical vs [61, 291] horizontal
  const mouthV = dist2D(landmarks[13], landmarks[14], imgW, imgH);
  const mouthH = Math.max(1, dist2D(landmarks[61], landmarks[291], imgW, imgH));
  const marValue = Number((mouthV / mouthH).toFixed(3));

  const hasBlendshapes = Array.isArray(blendshapes) && blendshapes.length > 0;
  const blinkAvg =
    (getBlendshapeScore(blendshapes, 'eyeBlinkLeft') + getBlendshapeScore(blendshapes, 'eyeBlinkRight')) / 2;
  const jawOpen = getBlendshapeScore(blendshapes, 'jawOpen');

  // Ground-truth calibrated for real webcams & varied classroom lighting (including mobile/phone cameras):
  // When closing eyes naturally or nodding off:
  // - Clear closure: blinkAvg >= 0.28 or earValue < 0.21
  // - Drowsy droop / slow blink: blinkAvg >= 0.22 combined with earValue < 0.24
  // - Fallback without blendshapes: earValue > 0 && earValue < 0.225 (guards against unpopulated 0-distance synthetic landmarks)
  const eyesClosed = hasBlendshapes
    ? blinkAvg >= 0.28 || (earValue > 0 && earValue < 0.21) || (blinkAvg >= 0.22 && earValue < 0.24)
    : earValue > 0 && earValue < 0.225;
  const isYawning = hasBlendshapes ? jawOpen > 0.40 || marValue > 0.46 : marValue > 0.46;

  return { earValue, marValue, eyesClosed, isYawning };
}

export function estimateGazeDirection(
  landmarks: NormalizedLandmark[],
  yawDeg: number,
  pitchDeg: number,
  blendshapes?: BlendshapeCategory[],
  faceQuality = 1
): BehavioralFeatures['gazeDirection'] {
  if (faceQuality < 0.35 || !landmarks || landmarks.length < 468) {
    return 'Unable to Determine';
  }

  // Head-pose-compensated horizontal & vertical gaze:
  // If blendshapes are present, combine neural eye-look offset with head yaw/pitch
  if (blendshapes && blendshapes.length > 0) {
    const lookLeft =
      (getBlendshapeScore(blendshapes, 'eyeLookOutLeft') + getBlendshapeScore(blendshapes, 'eyeLookInRight')) / 2;
    const lookRight =
      (getBlendshapeScore(blendshapes, 'eyeLookInLeft') + getBlendshapeScore(blendshapes, 'eyeLookOutRight')) / 2;
    const lookDown =
      (getBlendshapeScore(blendshapes, 'eyeLookDownLeft') + getBlendshapeScore(blendshapes, 'eyeLookDownRight')) / 2;

    // Compensate head yaw (normalized to [-1..1] around 35 deg) with opposite eye rotation
    const compensatedHoriz = yawDeg / 35 + (lookRight - lookLeft) * 0.85;
    const compensatedDown = pitchDeg / 30 + Math.max(0, lookDown - 0.2) * 0.8;

    if (compensatedHoriz < -0.68) return 'Looking Left';
    if (compensatedHoriz > 0.68) return 'Looking Right';
    if (compensatedDown > 0.72 || pitchDeg > 24) return 'Looking Down / Away';
    return 'Looking at Board';
  }

  if (Math.abs(yawDeg) > 24) {
    return yawDeg < 0 ? 'Looking Left' : 'Looking Right';
  }
  if (pitchDeg > 22) {
    return 'Looking Down / Away';
  }

  // Fallback: average both iris horizontal ratios (468 left, 473 right)
  if (landmarks.length >= 478) {
    const lInner = landmarks[133].x;
    const lOuter = landmarks[33].x;
    const lIris = landmarks[468].x;
    const lSpan = Math.max(1e-4, Math.abs(lInner - lOuter));
    const lRatio = (lIris - Math.min(lInner, lOuter)) / lSpan;

    const rInner = landmarks[362].x;
    const rOuter = landmarks[263].x;
    const rIris = landmarks[473].x;
    const rSpan = Math.max(1e-4, Math.abs(rOuter - rInner));
    const rRatio = (rIris - Math.min(rInner, rOuter)) / rSpan;

    const avgRatio = (lRatio + rRatio) / 2;
    if (avgRatio < 0.22) return 'Looking Right';
    if (avgRatio > 0.78) return 'Looking Left';
  }

  return 'Looking at Board';
}

export function classifyExpressionFromBlendshapes(
  blendshapes?: BlendshapeCategory[],
  eyesClosed = false
): {
  emotion: BehavioralFeatures['emotion'];
  confidence: number;
  probabilities: Record<string, number>;
  valid: boolean;
  rejectionReason: string | null;
} {
  if (!blendshapes || blendshapes.length === 0) {
    return {
      emotion: 'Unable to Determine',
      confidence: 0,
      probabilities: {},
      valid: false,
      rejectionReason: 'Facial blendshapes unavailable',
    };
  }

  const smile =
    (getBlendshapeScore(blendshapes, 'mouthSmileLeft') + getBlendshapeScore(blendshapes, 'mouthSmileRight')) / 2;
  const browDown =
    (getBlendshapeScore(blendshapes, 'browDownLeft') + getBlendshapeScore(blendshapes, 'browDownRight')) / 2;
  const browInnerUp = getBlendshapeScore(blendshapes, 'browInnerUp');
  const frown =
    (getBlendshapeScore(blendshapes, 'mouthFrownLeft') + getBlendshapeScore(blendshapes, 'mouthFrownRight')) / 2;
  const blink =
    (getBlendshapeScore(blendshapes, 'eyeBlinkLeft') + getBlendshapeScore(blendshapes, 'eyeBlinkRight')) / 2;
  const jawOpen = getBlendshapeScore(blendshapes, 'jawOpen');

  if (eyesClosed || blink >= 0.32 || (jawOpen > 0.40 && blink >= 0.22)) {
    return {
      emotion: 'Sleepy',
      confidence: Number(Math.min(0.96, 0.72 + Math.max(blink, jawOpen) * 0.26).toFixed(2)),
      probabilities: {
        Sleepy: 0.85,
        Neutral: 0.08,
        Confused: 0.03,
        Frustrated: 0.02,
        Happy: 0.02,
      },
      valid: true,
      rejectionReason: null,
    };
  }
  const rawLogits: Record<'Neutral' | 'Happy' | 'Confused' | 'Sleepy' | 'Frustrated', number> = {
    Neutral: 1.15 - (smile * 1.4 + browDown * 1.2 + frown * 1.2 + (eyesClosed ? 1.8 : blink * 0.9)),
    Happy: smile * 3.1 - frown * 0.8,
    Confused: browDown * 1.8 + browInnerUp * 2.1 - smile * 0.9,
    Sleepy: (eyesClosed ? 2.4 : blink * 2.5) + jawOpen * 1.1 - smile * 0.6,
    Frustrated: frown * 2.2 + browDown * 1.4 - smile * 1.0,
  };

  const entries = Object.entries(rawLogits) as Array<[keyof typeof rawLogits, number]>;
  const maxLogit = Math.max(...entries.map(([, v]) => v));
  const exps = entries.map(([k, v]) => [k, Math.exp(v - maxLogit)] as const);
  const sumExp = exps.reduce((acc, [, e]) => acc + e, 0) || 1;
  const probabilities: Record<string, number> = {};
  let bestEmotion: BehavioralFeatures['emotion'] = 'Neutral';
  let bestProb = 0;

  for (const [k, e] of exps) {
    const p = Number((e / sumExp).toFixed(3));
    probabilities[k] = p;
    if (p > bestProb) {
      bestProb = p;
      bestEmotion = k;
    }
  }

  return {
    emotion: bestEmotion,
    confidence: Number(bestProb.toFixed(2)),
    probabilities,
    valid: true,
    rejectionReason: null,
  };
}

export function extractPoseBehavior(
  poseLandmarks?: NormalizedLandmark[] | null
): {
  handRaised: boolean;
  posture: BehavioralFeatures['posture'];
  postureConfidence: number;
  handConfidence: number;
  postureRejection: string | null;
  handRejection: string | null;
} {
  if (!poseLandmarks || poseLandmarks.length < 25) {
    return {
      handRaised: false,
      posture: 'Unknown',
      postureConfidence: 0,
      handConfidence: 0,
      postureRejection: 'Pose landmarks unavailable',
      handRejection: 'Pose landmarks unavailable',
    };
  }

  const nose = poseLandmarks[0];
  const ls = poseLandmarks[11];
  const rs = poseLandmarks[12];
  const le = poseLandmarks[13];
  const re = poseLandmarks[14];
  const lw = poseLandmarks[15];
  const rw = poseLandmarks[16];
  const lh = poseLandmarks[23];
  const rh = poseLandmarks[24];
  const lk = poseLandmarks[25];
  const rk = poseLandmarks[26];

  let handRaised = false;
  let handConfidence = 0;
  let handRejection: string | null = null;
  const leftArmVis = Math.min(ls?.visibility ?? 0, lw?.visibility ?? 0);
  const rightArmVis = Math.min(rs?.visibility ?? 0, rw?.visibility ?? 0);

  if (leftArmVis <= 0.4 && rightArmVis <= 0.4) {
    handRejection = 'Shoulder/wrist landmarks occluded or low visibility';
  } else {
    const leftElbowCheck = (le?.visibility ?? 0) > 0.3 ? le.y < ls.y + 0.14 : true;
    const rightElbowCheck = (re?.visibility ?? 0) > 0.3 ? re.y < rs.y + 0.14 : true;
    if ((ls?.visibility ?? 0) > 0.4 && (lw?.visibility ?? 0) > 0.45 && lw.y < ls.y - 0.05 && leftElbowCheck) {
      handRaised = true;
    }
    if ((rs?.visibility ?? 0) > 0.4 && (rw?.visibility ?? 0) > 0.45 && rw.y < rs.y - 0.05 && rightElbowCheck) {
      handRaised = true;
    }
    handConfidence = Number(Math.max(leftArmVis, rightArmVis).toFixed(2));
  }

  let posture: BehavioralFeatures['posture'] = 'Unknown';
  let postureConfidence = 0;
  let postureRejection: string | null = 'Shoulders not clearly visible';
  if ((ls?.visibility ?? 0) > 0.35 && (rs?.visibility ?? 0) > 0.35) {
    postureRejection = null;
    postureConfidence = Number((((ls.visibility ?? 0.5) + (rs.visibility ?? 0.5)) / 2).toFixed(2));
    const shoulderMidY = (ls.y + rs.y) / 2;
    const shoulderTilt = Math.abs(ls.y - rs.y);
    const hipVisible = (lh?.visibility ?? 0) > 0.3 && (rh?.visibility ?? 0) > 0.3;
    const hipMidY = hipVisible ? (lh.y + rh.y) / 2 : shoulderMidY + 0.35;
    const kneeVisible = (lk?.visibility ?? 0) > 0.45 && (rk?.visibility ?? 0) > 0.45;
    const kneeMidY = kneeVisible ? (lk.y + rk.y) / 2 : 0;
    const torsoDrop = (nose?.visibility ?? 0) > 0.4 ? shoulderMidY - nose.y : 0.14;

    if (shoulderTilt > 0.07 || Math.abs(hipMidY - shoulderMidY) < 0.18) {
      posture = 'Slouching';
    } else if (torsoDrop < 0.08) {
      posture = 'Leaning';
    } else if (hipVisible && kneeVisible && kneeMidY - hipMidY > 0.18 && shoulderMidY < 0.45) {
      posture = 'Standing';
    } else {
      posture = 'Upright';
    }
  }

  return { handRaised, posture, postureConfidence, handConfidence, postureRejection, handRejection };
}

/**
 * Analyzes photometric screen glow reflection on the student's face.
 * Detects upward diffuse screen illumination (higher chin luminance vs forehead / cool blue light spectrum)
 * cast by mobile phones and hand-held screens held below eye level.
 */
export function estimateScreenGlowFromPixels(
  foreheadRgb: [number, number, number],
  chinRgb: [number, number, number],
  eyeLookDownAvg: number,
  pitchDeg: number
): { detected: boolean; confidence: number; ratio: number } {
  const [fR, fG, fB] = foreheadRgb;
  const [cR, cG, cB] = chinRgb;

  const foreLum = 0.299 * fR + 0.587 * fG + 0.114 * fB;
  const chinLum = 0.299 * cR + 0.587 * cG + 0.114 * cB;

  const lumRatio = chinLum / Math.max(1.0, foreLum);
  const chinBlueRatio = cB / Math.max(1.0, cR);
  const foreBlueRatio = fB / Math.max(1.0, fR);
  const isCoolTint = chinBlueRatio > foreBlueRatio + 0.12;

  // Upward screen illumination: chin is noticeably brighter than forehead (normal ambient lighting has forehead brighter)
  const isUpwardLight = lumRatio >= 1.10;
  const isLookingDownAtScreen = eyeLookDownAvg >= 0.22 || pitchDeg >= 7;

  if ((isUpwardLight || (isCoolTint && chinLum >= 45)) && isLookingDownAtScreen) {
    const conf = Math.min(0.96, Math.max(0.65, (lumRatio - 1.0) * 1.5 + (isCoolTint ? 0.2 : 0)));
    return { detected: true, confidence: Number(conf.toFixed(2)), ratio: Number(lumRatio.toFixed(2)) };
  }

  return { detected: false, confidence: 0, ratio: Number(lumRatio.toFixed(2)) };
}

export function extractBehavioralFeatures(
  faceLandmarks: NormalizedLandmark[] | null,
  blendshapes: BlendshapeCategory[] | undefined,
  poseLandmarks: NormalizedLandmark[] | null | undefined,
  faceQuality: number,
  imgW: number,
  imgH: number,
  phoneDetected = false,
  transformMatrix?: number[] | Float32Array,
  screenGlowDetected = false,
  screenGlowConfidence = 0.0
): BehavioralFeatures {
  const ts = Date.now();
  const poseBehavior = extractPoseBehavior(poseLandmarks);
  const postureModality: ModalityObservation = {
    observation: poseBehavior.posture,
    confidence: poseBehavior.postureConfidence,
    valid: poseBehavior.posture !== 'Unknown',
    timestamp: ts,
    rejectionReason: poseBehavior.postureRejection,
  };
  const handModality: ModalityObservation = {
    observation: poseBehavior.handRaised ? 'Raised' : 'Lowered',
    confidence: poseBehavior.handConfidence,
    valid: poseBehavior.handRejection === null,
    timestamp: ts,
    rejectionReason: poseBehavior.handRejection,
  };

  if (!faceLandmarks || faceLandmarks.length < 468) {
    return {
      faceDetected: false,
      headPoseLabel: 'Unknown',
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      gazeDirection: 'Unable to Determine',
      earValue: 0,
      marValue: 0,
      eyesClosed: false,
      isYawning: false,
      eyeLookDownAvg: 0,
      gazeDeviation: 0,
      phoneDetected,
      screenGlowDetected: false,
      screenGlowConfidence: 0,
      emotion: 'Unable to Determine',
      emotionConfidence: 0,
      expressionProbabilities: {},
      handRaised: poseBehavior.handRaised,
      posture: poseBehavior.posture,
      faceQuality: 0,
      modalities: {
        gaze: { observation: 'Unable to Determine', confidence: 0, valid: false, timestamp: ts, rejectionReason: 'Face landmarks missing' },
        headPose: { observation: 'Unknown', confidence: 0, valid: false, timestamp: ts, rejectionReason: 'Face landmarks missing' },
        expression: { observation: 'Unable to Determine', confidence: 0, valid: false, timestamp: ts, rejectionReason: 'Face landmarks missing' },
        posture: postureModality,
        handRaise: handModality,
        eyeClosure: { observation: 'Unknown', confidence: 0, valid: false, timestamp: ts, rejectionReason: 'Face landmarks missing' },
      },
    };
  }

  const head =
    transformMatrix && transformMatrix.length >= 12
      ? estimateHeadPoseFromMatrix(transformMatrix)
      : estimateHeadPoseFromLandmarks(faceLandmarks, imgW, imgH);
  const eyeMouth = estimateEyeAndMouth(faceLandmarks, imgW, imgH, blendshapes);
  const gazeDirection = estimateGazeDirection(faceLandmarks, head.yawDeg, head.pitchDeg, blendshapes, faceQuality);
  const expr = classifyExpressionFromBlendshapes(blendshapes, eyeMouth.eyesClosed);
  const eyeLookDownAvg = Number(
    ((getBlendshapeScore(blendshapes, 'eyeLookDownLeft') + getBlendshapeScore(blendshapes, 'eyeLookDownRight')) / 2).toFixed(2)
  );
  const lookLeft =
    (getBlendshapeScore(blendshapes, 'eyeLookOutLeft') + getBlendshapeScore(blendshapes, 'eyeLookInRight')) / 2;
  const lookRight =
    (getBlendshapeScore(blendshapes, 'eyeLookInLeft') + getBlendshapeScore(blendshapes, 'eyeLookOutRight')) / 2;
  const gazeDeviation = Number(
    Math.min(1, Math.max(Math.abs(lookLeft - lookRight), Math.abs(head.yawDeg) / 45, Math.max(0, eyeLookDownAvg - 0.35))).toFixed(2)
  );

  const gazeValid = gazeDirection !== 'Unable to Determine' && faceQuality >= 0.35;
  const headValid = faceQuality >= 0.35;
  const eyeValid = faceQuality >= 0.35 && eyeMouth.earValue > 0;

  return {
    faceDetected: true,
    headPoseLabel: head.label,
    pitchDeg: head.pitchDeg,
    yawDeg: head.yawDeg,
    rollDeg: head.rollDeg,
    gazeDirection,
    earValue: eyeMouth.earValue,
    marValue: eyeMouth.marValue,
    eyesClosed: eyeMouth.eyesClosed,
    isYawning: eyeMouth.isYawning,
    eyeLookDownAvg,
    gazeDeviation,
    phoneDetected,
    screenGlowDetected,
    screenGlowConfidence,
    emotion: expr.emotion,
    emotionConfidence: expr.confidence,
    expressionProbabilities: expr.probabilities,
    handRaised: poseBehavior.handRaised,
    posture: poseBehavior.posture,
    faceQuality,
    modalities: {
      gaze: {
        observation: gazeDirection,
        confidence: gazeValid ? Number(faceQuality.toFixed(2)) : 0,
        valid: gazeValid,
        timestamp: ts,
        rejectionReason: gazeValid ? null : 'Low facial landmark quality for gaze estimation',
      },
      headPose: {
        observation: head.label,
        confidence: headValid ? Number(faceQuality.toFixed(2)) : 0,
        valid: headValid,
        timestamp: ts,
        rejectionReason: headValid ? null : 'Low facial landmark quality for head pose',
      },
      expression: {
        observation: expr.emotion,
        confidence: expr.confidence,
        valid: expr.valid,
        timestamp: ts,
        rejectionReason: expr.rejectionReason,
      },
      posture: postureModality,
      handRaise: handModality,
      eyeClosure: {
        observation: eyeMouth.eyesClosed ? 'Closed' : 'Open',
        confidence: eyeValid ? Number(Math.min(0.95, 0.55 + faceQuality * 0.4).toFixed(2)) : 0,
        valid: eyeValid,
        timestamp: ts,
        rejectionReason: eyeValid ? null : 'Eye landmarks unreliable',
      },
    },
  };
}

