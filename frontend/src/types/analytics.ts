export interface StudentComponents {
  attention_contribution: number;
  emotion_contribution: number;
  participation_contribution: number;
  posture_contribution: number;
  drowsiness_penalty: number;
}

export interface StudentDebugInfo {
  student_id: string;
  face_detected?: boolean;
  valid_face_count?: number;
  detected_person_count?: number;
  active_track_count?: number;
  raw_model_output?: number;
  raw_feature_values?: {
    gaze?: string;
    head_pose?: string;
    posture?: string;
    ear?: number | null;
    face_quality?: number;
    emotion?: string;
    drowsy_evidence_frame?: boolean;
    hand_raised_frame?: boolean;
    pitch_deg?: number | null;
    yaw_deg?: number | null;
  };
  temporal_values?: {
    raw_attention_score?: number;
    smoothed_attention_score?: number;
    drowsy_window_count?: number;
    drowsy_window_total?: number;
    drowsy_window_ratio_pct?: number;
    is_drowsy_consistent?: boolean;
  };
  reliability_values?: {
    face_quality?: number;
    confidence_pct?: number;
    is_valid_evidence?: boolean;
  };
  final_attention_score: number;
  final_state: string;
  reason_for_state: string;
}

export interface StudentAnalytics {
  student_id: string;
  track_id?: number;
  bbox?: number[] | null;
  person_bbox?: number[];
  face_detected?: boolean;
  face_status?: string;
  confidence_pct?: number;
  face_quality?: number;
  attention_pct: number;
  attention_status?: string;
  final_state?: string;
  cautious_engagement_category?:
    | 'Attentive-related behaviour estimate'
    | 'Partially engaged or uncertain'
    | 'Low visible engagement-related cues'
    | 'Insufficient evidence';
  reason_for_state?: string;
  gaze_direction: string;
  gaze_vector?: string;
  head_pose?: string;
  pitch_deg?: number | null;
  yaw_deg?: number | null;
  roll_deg?: number | null;
  emotion: string;
  expression_probabilities?: Record<string, number>;
  modality_observations?: Record<
    string,
    {
      observation: string;
      confidence: number;
      valid: boolean;
      timestamp: number;
      rejectionReason: string | null;
    }
  >;
  is_drowsy: boolean;
  is_yawning: boolean;
  hand_raised: boolean;
  posture: string;
  pose_visible?: boolean;
  participation_pct: number;
  engagement_pct: number;
  confusion_index: number;
  confusion_level: string;
  esi_score: number;
  status: string;
  ear_value?: number | null;
  mar_value?: number | null;
  perclos_pct?: number;
  blink_rate_bpm?: number;
  phone_detected?: boolean;
  behavioral_cue?: string;
  ai_suggestion?: {
    action: string;
    pedagogical_tip: string;
    priority: 'HIGH' | 'MEDIUM' | 'LOW';
  };
  corrective_guidance?: {
    is_fallback: boolean;
    reason?: string;
    action_prompt: string;
    target_accuracy_path: string;
  };
  components?: StudentComponents;
  modality_reliabilities?: Record<string, number>;
  debug_info?: StudentDebugInfo;
  xai?: {
    attention?: { reasons: string[]; confidence_pct?: number; status?: string };
    engagement?: { reasons?: string[]; summary_rationale?: string; attributions?: string[] };
    confusion?: { triggers?: string[]; primary_driver?: string };
    drowsiness?: { diagnostic_markers?: string[] };
    counterfactual?: { explanation?: string; [key: string]: any };
  };
}

export interface TeacherRecommendation {
  id: string;
  timestamp: string;
  type: string;
  severity: 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO';
  title: string;
  message: string;
  action: string;
}

export interface TimelinePoint {
  time: string;
  attention: number;
  engagement: number;
  confusion: number;
}

export interface ClassroomMetrics {
  eci_score: number;
  eci_status: string;
  avg_attention: number;
  avg_engagement: number;
  avg_participation: number;
  avg_confusion: number;
  drowsy_count: number;
  hand_raise_count: number;
  phone_distraction_count?: number;
  live_detected_persons?: number;
  valid_face_count?: number;
  active_tracks?: number;
  camera_connected?: boolean;
  camera_source?: string;
  performance?: {
    fps: number;
    latency_ms: number;
  };
  students?: StudentAnalytics[];
  recommendations?: TeacherRecommendation[];
}
