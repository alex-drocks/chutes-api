"""
Challenge schemas.
"""

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    func,
    BigInteger,
)
from sqlalchemy.orm import relationship
from api.database import Base


class Challenge(Base):
    __tablename__ = "miner_challenges"
    uuid = Column(String, ForeignKey("nodes.uuid", ondelete="CASCADE"), primary_key=True)
    seed = Column(BigInteger, primary_key=True, default=0)
    challenge = Column(String, nullable=False)
    challenge_type = Column(String, default="graval", primary_key=True)
    created_at = Column(DateTime, server_default=func.now())

    node = relationship("Node", back_populates="challenges")
