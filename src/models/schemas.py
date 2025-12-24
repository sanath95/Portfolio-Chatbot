"""Pydantic models for agent communication and data structures."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DownstreamAgent(str, Enum):
    """Available downstream agents for routing."""
    PROFESSIONAL_INFO = "professional_info"
    PUBLIC_PERSONA = "public_persona"
    FINAL_PRESENTATION = "final_presentation"


class Support(str, Enum):
    """Sources of evidence support."""
    RESUME = "resume"
    ABOUT_SANATH = "about_sanath"
    RETRIEVE = "retrieve"


class Reinterpretation(BaseModel):
    """Query reinterpretation result.
    
    Attributes:
        needed: Whether reinterpretation was required.
        rewritten_question: Rewritten user query if reinterpretation was needed.
    """
    needed: bool = Field(..., description="Whether reinterpretation was required")
    rewritten_question: Optional[str] = Field(
        None,
        description="Rewritten user query if reinterpretation was needed",
    )


class DownstreamRequest(BaseModel):
    """Request to a downstream agent.
    
    Attributes:
        agent: Target agent identifier.
        task: What the agent is supposed to do.
        constraints: Key do/don't rules specific to this request.
    """
    agent: DownstreamAgent
    task: str = Field(..., description="What the agent is supposed to do")
    constraints: Optional[str] = Field(
        None,
        description="Key do/don't rules specific to this request",
    )


class RefusalDirective(BaseModel):
    """Directive for refusing a request.
    
    Attributes:
        needed: Whether a refusal is required.
        reason: Why refusal applies.
        style: How to style the refusal.
    """
    needed: bool = Field(..., description="Whether a refusal is required")
    reason: Optional[str] = Field(None, description="Why refusal applies")
    style: Optional[str] = Field(default="polite and humorous with redirect")


class OrchestratorRoute(BaseModel):
    """Orchestrator routing decision.
    
    Attributes:
        reinterpretation: Query reinterpretation result.
        downstream_requests: List of requests to downstream agents.
        refusal_directive: Refusal directive if applicable.
    """
    reinterpretation: Reinterpretation
    downstream_requests: list[DownstreamRequest]
    refusal_directive: RefusalDirective


class CoverageAssessment(BaseModel):
    """Assessment of evidence coverage.
    
    Attributes:
        sufficient: Whether evidence is sufficient.
        missing_points: Specific aspects lacking evidence.
    """
    sufficient: bool
    missing_points: list[str] = Field(
        ...,
        description="Specific aspects of the user's query that lack evidence",
    )


class Claims(BaseModel):
    """Evidence claims with support.
    
    Attributes:
        documents: Retrieved documents or text chunks.
        support: Source of support for the claims.
    """
    documents: list[str] = Field(
        ..., description="Full length documents or text chunks retrieved from tools"
    )
    support: Support


class EvidenceBundle(BaseModel):
    """Bundle of evidence for answering a query.
    
    Attributes:
        coverage_assessment: Assessment of evidence coverage.
        claims: List of claims with supporting documents.
        project_leads: Relevant projects, if evidenced.
        safe_redirect_if_missing: Redirect suggestion if evidence is missing.
    """
    coverage_assessment: CoverageAssessment
    claims: list[Claims]
    project_leads: Optional[list[str]] = Field(
        None, description="Projects that appear relevant, if evidenced"
    )
    safe_redirect_if_missing: Optional[str] = Field(
        None,
        description="One sentence the Final Presentation agent can use if evidence is missing",
    )