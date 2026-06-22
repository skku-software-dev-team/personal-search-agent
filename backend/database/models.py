from database.database import Base
from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.sql import func


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # google_id
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    picture = Column(String, nullable=True)
    access_token = Column(Text, nullable=True)  # 암호화 저장
    refresh_token = Column(Text, nullable=True)  # 암호화 저장
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
