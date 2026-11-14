import { describe, expect, it } from 'vitest';
import { DATASET_SAMPLES } from '../../components/LiveClassroomView';
import { isPhoneAssociatedWithPerson, isInsideDistractorScreen } from '../validation';
import { BrowserByteTracker, TrackedPerson } from '../tracker';
import { BrowserEngagementFusion } from '../fusion';
import { BehavioralFeatures } from '../features';
import { drawOverlayBoxes } from '../detector';

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
    ...overrides,
  };
}

describe('Dataset Samples & Phone Distraction Feature End-to-End Suite', () => {
  it('1. Verifies that all 4 free open dataset samples are configured with valid paths and badges', () => {
    expect(DATASET_SAMPLES).toHaveLength(4);

    const ids = DATASET_SAMPLES.map((s) => s.id);
    expect(ids).toContain('phone-distraction');
    expect(ids).toContain('classroom-group');
    expect(ids).toContain('phone-laptop-study');
    expect(ids).toContain('attentive-student');

    for (const sample of DATASET_SAMPLES) {
      expect(sample.url).toMatch(/^\/samples\/.+\.jpg$/);
      expect(sample.name.length).toBeGreaterThan(0);
      expect(sample.badge.length).toBeGreaterThan(0);
      expect(sample.badgeColor).toContain('border-');
    }

    const phoneSample = DATASET_SAMPLES.find((s) => s.id === 'phone-distraction');
    expect(phoneSample?.badge).toContain('PHONE DISTRACTED');
    expect(phoneSample?.badgeColor).toContain('purple');
  });

  it('2. Tests spatial phone association against student bounding boxes', () => {
    const studentBox = { x1: 100, y1: 50, x2: 300, y2: 450, score: 0.9, label: 'person' };
    const distantStudent = { x1: 500, y1: 50, x2: 700, y2: 450, score: 0.9, label: 'person' };

    // Phone held by first student
    const activePhone = [{ x1: 180, y1: 280, x2: 240, y2: 380, score: 0.85, label: 'cell phone' }];
    expect(isPhoneAssociatedWithPerson(studentBox, activePhone)).toBe(true);
    expect(isPhoneAssociatedWithPerson(distantStudent, activePhone)).toBe(false);

    // Laptop distractor should NOT trigger phone association
    const laptopDistractor = [{ x1: 180, y1: 280, x2: 280, y2: 380, score: 0.85, label: 'laptop' }];
    expect(isPhoneAssociatedWithPerson(studentBox, laptopDistractor)).toBe(false);
  });

  it('3. Verifies that phone detection forces PHONE DISTRACTED state, drops attention, and raises priority alert', () => {
    const fusion = new BrowserEngagementFusion();
    const track = createMockTrack('Student 01', 1);

    const baseFeatures = createBaseFeatures({
      phoneDetected: true, // Ground truth from COCO phone detection!
    });

    const studentAnalytics = fusion.evaluateStudent(track, baseFeatures);

    expect(studentAnalytics.final_state).toBe('PHONE DISTRACTED');
    expect(studentAnalytics.phone_detected).toBe(true);
    expect(studentAnalytics.behavioral_cue).toBe('Phone Distraction');
    expect(studentAnalytics.attention_status).toBe('Phone / Secondary Device In Use');
    expect(studentAnalytics.attention_pct).toBeLessThanOrEqual(30); // Calibrated low attention

    expect(studentAnalytics.ai_suggestion).toBeDefined();
    expect(studentAnalytics.ai_suggestion?.action).toBe('Device Distraction Redirection');
    expect(studentAnalytics.ai_suggestion?.priority).toBe('HIGH');
    expect(studentAnalytics.ai_suggestion?.pedagogical_tip).toContain('Mobile phone detected');
  });

  it('4. Aggregates classroom metrics and increments phone_distraction_count', () => {
    const fusion = new BrowserEngagementFusion();
    const track1 = createMockTrack('Student 01', 1);
    const track2 = createMockTrack('Student 02', 2);

    const student1 = fusion.evaluateStudent(
      track1,
      createBaseFeatures({
        phoneDetected: false,
      })
    );

    const student2 = fusion.evaluateStudent(
      track2,
      createBaseFeatures({
        gazeDirection: 'Looking Down / Away',
        gazeDeviation: 0.25,
        pitchDeg: 12,
        yawDeg: 10,
        headPoseLabel: 'Looking Down',
        posture: 'Slouching',
        phoneDetected: true, // Distracted student
      })
    );

    const classroom = fusion.aggregateClassroom([student1, student2], 2, 2, 10.0, 15.0, true);

    expect(classroom.phone_distraction_count).toBe(1);
    expect(classroom.valid_face_count).toBe(2);
    expect(classroom.live_detected_persons).toBe(2);

    // Verifies that a specific recommendation for phone distraction is generated
    const phoneRec = classroom.recommendations?.find((r) => r.id.startsWith('rec-phone'));
    expect(phoneRec).toBeDefined();
    expect(phoneRec?.severity).toBe('HIGH');
    expect(phoneRec?.title).toBe('Secondary Device In Use');
  });

  it('5. Verifies canvas overlay rendering handles PHONE DISTRACTED with purple badge', () => {
    // Mock 2D canvas context
    const filledRects: any[] = [];
    const filledTexts: any[] = [];
    const strokedRects: any[] = [];

    const mockCtx = {
      clearRect: () => {},
      strokeRect: (x: number, y: number, w: number, h: number) => strokedRects.push({ x, y, w, h }),
      fillRect: (x: number, y: number, w: number, h: number) => filledRects.push({ x, y, w, h }),
      fillText: (text: string, x: number, y: number) => filledTexts.push({ text, x, y }),
      measureText: (text: string) => ({ width: text.length * 8 }),
      font: '',
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
    };

    const mockCanvas = {
      width: 640,
      height: 480,
      getContext: () => mockCtx,
    } as unknown as HTMLCanvasElement;

    const student = {
      student_id: 'Student 01',
      final_state: 'PHONE DISTRACTED',
      attention_pct: 22,
      face_detected: true,
      bbox: [100, 100, 200, 200],
      person_bbox: [80, 80, 250, 450],
      gaze_direction: 'Looking Down',
      ai_suggestion: { action: 'Device Distraction Redirection' },
    } as any;

    drawOverlayBoxes(mockCanvas, 640, 480, [student]);

    expect(strokedRects.length).toBeGreaterThan(0);
    // Find text rendered with phone icon and state
    const phoneLabel = filledTexts.find((t) => t.text.includes('PHONE DISTRACTED') && t.text.includes('📱'));
    expect(phoneLabel).toBeDefined();
    expect(phoneLabel.text).toContain('Student 01');
    expect(phoneLabel.text).toContain('22%');
  });
});
