from sqlalchemy import Column, DateTime, Integer, JSON, String, func

from database.connection import Base


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    analysis_id = Column(String(32), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    media_type = Column(String(20), nullable=False)
    file_hash = Column(String(64), nullable=False)
    bytes_received = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)
    response_json = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )