# EduSense Entity-Relationship (ER) Diagram Specification

## Overview
The EduSense database schema is designed for time-series classroom analytics logging and anonymous student session management.

## Mermaid ER Diagram

```mermaid
erDiagram
    CLASSROOM_SESSION ||--o{ FRAME_ANALYTICS : contains
    CLASSROOM_SESSION ||--o{ ALERT_LOG : generates
    STUDENT ||--o{ FRAME_ANALYTICS : logs

    CLASSROOM_SESSION {
        int id PK
        string session_code UK
        string title
        datetime start_time
        datetime end_time
        boolean is_active
        float eci_score
        float avg_attention
        float avg_engagement
        float avg_participation
    }

    STUDENT {
        int id PK
        string anonymous_id UK "Student 01, Student 02..."
        int grid_row
        int grid_col
        float current_esi
        string status "Excellent, Good, Needs Attention"
        datetime created_at
    }

    FRAME_ANALYTICS {
        int id PK
        int session_id FK
        int student_id FK
        datetime timestamp
        float attention_pct
        string gaze_direction
        string emotion
        float emotion_confidence
        boolean is_drowsy
        float ear_value
        boolean is_yawning
        boolean hand_raised
        string posture_status
        float participation_pct
        float engagement_pct
        float confusion_index
        float esi_score
    }

    ALERT_LOG {
        int id PK
        int session_id FK
        datetime timestamp
        string alert_type "ATTENTION_DROP, CONFUSION_SPIKE..."
        string severity "HIGH, MEDIUM, LOW"
        string message
        string recommended_action
        boolean is_resolved
    }
```
