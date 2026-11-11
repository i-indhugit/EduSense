"""
EduSense Database Models & Connection Setup
============================================
SQLAlchemy ORM models for anonymous student analytics, session tracking,
time-series frame logs, and teacher recommendations.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "edusense.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ClassroomSession(Base):
    __tablename__ = "classroom_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_code = Column(String(50), unique=True, index=True, default="CLASS-LIVE-01")
    title = Column(String(100), default="Live Computer Science Lecture")
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    eci_score = Column(Float, default=85.0)
    avg_attention = Column(Float, default=82.5)
    avg_engagement = Column(Float, default=78.0)
    avg_participation = Column(Float, default=70.0)
    
    analytics = relationship("FrameAnalytics", back_populates="session", cascade="all, delete-orphan")
    alerts = relationship("AlertLog", back_populates="session", cascade="all, delete-orphan")

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    anonymous_id = Column(String(50), unique=True, index=True)  # e.g., "Student 01"
    grid_row = Column(Integer, default=1)
    grid_col = Column(Integer, default=1)
    current_esi = Column(Float, default=85.0)
    status = Column(String(20), default="Good")  # Excellent, Good, Needs Attention
    created_at = Column(DateTime, default=datetime.utcnow)

    analytics = relationship("FrameAnalytics", back_populates="student", cascade="all, delete-orphan")

class FrameAnalytics(Base):
    __tablename__ = "frame_analytics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("classroom_sessions.id"))
    student_id = Column(Integer, ForeignKey("students.id"))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    attention_pct = Column(Float, default=85.0)
    gaze_direction = Column(String(30), default="Looking at Board") # Looking at Board, Looking Away, Head Down
    emotion = Column(String(30), default="Neutral")               # Neutral, Happy, Confused, Bored, Sleepy, Frustrated
    emotion_confidence = Column(Float, default=0.90)
    
    is_drowsy = Column(Boolean, default=False)
    ear_value = Column(Float, default=0.30)
    is_yawning = Column(Boolean, default=False)
    
    hand_raised = Column(Boolean, default=False)
    posture_status = Column(String(30), default="Upright")         # Upright, Leaning, Slouching
    
    participation_pct = Column(Float, default=75.0)
    engagement_pct = Column(Float, default=80.0)
    confusion_index = Column(Float, default=15.0)
    esi_score = Column(Float, default=82.0)

    session = relationship("ClassroomSession", back_populates="analytics")
    student = relationship("Student", back_populates="analytics")

class AlertLog(Base):
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("classroom_sessions.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String(50))   # ATTENTION_DROP, CONFUSION_SPIKE, DROWSINESS_CLUSTER, LOW_PARTICIPATION
    severity = Column(String(20))     # HIGH, MEDIUM, LOW
    message = Column(Text)
    recommended_action = Column(Text)
    is_resolved = Column(Boolean, default=False)

    session = relationship("ClassroomSession", back_populates="alerts")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
