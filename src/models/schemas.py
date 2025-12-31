"""Pydantic models for agent communication and data structures."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Literal, Union

from pydantic import BaseModel, Field


class DownstreamAgent(str, Enum):
    """Available downstream agents for routing."""
    PROFESSIONAL_INFO = "professional_info"
    PUBLIC_PERSONA = "public_persona"
    FINAL_PRESENTATION = "final_presentation"


class Support(str, Enum):
    """Sources of evidence support."""
    RESUME = "resume"
    OLD_RESUME = "old_resume"
    ABOUT_SANATH = "about_sanath"
    GITHUB = "github"
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
    
class AgentSource(str, Enum):
    ORCHESTRATOR = "orchestrator"
    PROFESSIONAL_INFO = "professional_info"
    PUBLIC_PERSONA = "public_persona"
    FINAL_PRESENTATION = "final_presentation"
    
class OrchestratorEvent(BaseModel):
    from_: Literal[AgentSource.ORCHESTRATOR]
    output: str

class ProfessionalInfoEvent(BaseModel):
    from_: Literal[AgentSource.PROFESSIONAL_INFO]
    output: str
    
class PublicPersonaEvent(BaseModel):
    from_: Literal[AgentSource.PUBLIC_PERSONA]
    output: str

class FinalPresentationEvent(BaseModel):
    from_: Literal[AgentSource.FINAL_PRESENTATION]
    output: str

AgentEventUnion = Union[
    OrchestratorEvent,
    ProfessionalInfoEvent,
    PublicPersonaEvent,
    FinalPresentationEvent,
]

class Platform(str, Enum):
    """Supported public content platforms."""
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"

class EngagementMetrics(BaseModel):
    """Platform-agnostic engagement metrics for a public artifact."""
    view_count: Optional[int] = Field(
        None,
        description="Number of views,",
    )
    like_count: Optional[int] = Field(
        None,
        description="Number of likes or positive reactions.",
    )
    comment_count: Optional[int] = Field(
        None,
        description="Number of comments or replies.",
    )
    
class Artifact(BaseModel):
    """A single public artifact created or curated by Sanath."""
    platform: Platform = Field(
        ...,
        description="Platform on which the artifact is published.",
    )
    title: str = Field(
        ...,
        description="Title or primary label of the artifact.",
    )
    description: Optional[str] = Field(
        None,
        description="Original platform-provided description (no interpretation).",
    )
    url: str = Field(
        ...,
        description="Direct public URL to the artifact.",
    )
    timestamp: str = Field(
        ...,
        description="Normalized publication timestamp (ISO).",
    )
    engagement_metrics: EngagementMetrics = Field(
        ...,
        description="Observed engagement metrics for the artifact.",
    )
    factual_description: Optional[str] = Field(
        None,
        description="One-sentence factual description of what the artifact is, without interpretation or inference.",
    )

class InstagramAccountMetadata(BaseModel):
    """Public metadata for Sanath's Instagram account."""
    followers_count: Optional[int] = Field(
        None,
        description="Number of followers on the Instagram account.",
    )
    profile_description: Optional[str] = Field(
        None,
        description="Account bio text as shown on Instagram.",
    )
    media_count: Optional[int] = Field(
        None,
        description="Total number of media items posted.",
    )
    
class YouTubeChannelMetadata(BaseModel):
    """Public metadata for Sanath's YouTube channel."""
    subscribers_count: Optional[int] = Field(
        None,
        description="Number of channel subscribers.",
    )
    profile_description: Optional[str] = Field(
        None,
        description="Channel description as shown on YouTube.",
    )
    video_count: Optional[int] = Field(
        None,
        description="Total number of uploaded videos.",
    )

class AccountMetadata(BaseModel):
    """Platform-level metadata for a public account."""
    handle: str = Field(
        ...,
        description="Platform-specific account identifier (e.g., username or channel ID).",
    )
    platform_metadata: Optional[
        Union[InstagramAccountMetadata, YouTubeChannelMetadata]
    ] = Field(
        None,
        description="Public metadata specific to the referenced platform.",
    )

class PublicArtifacts(BaseModel):
    """Complete evidence bundle returned by the Public Persona Evidence Agent."""
    coverage_assessment: CoverageAssessment = Field(
        ...,
        description="Assessment of whether the collected artifacts sufficiently cover the user's query.",
    )
    artifacts: list[Artifact] = Field(
        ...,
        description="List of verified public artifacts relevant to the query.",
    )
    account_metadata: Optional[list[AccountMetadata]] = Field(
        ...,
        description="Platform-level metadata associated with the collected artifacts.",
    )
    safe_redirect_if_missing: Optional[str] = Field(
        None,
        description="Single-sentence fallback response for downstream use when evidence is insufficient.",
    )